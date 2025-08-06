import gc
import os
import numpy as np
import pandas as pd
from itertools import compress
from v1dd_public import ARTIFACT_DIR
from sklearn import metrics, preprocessing
from allen_v1dd.client import OPhysClient

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import logging
import warnings
from datetime import datetime
from multiprocessing import Manager
from joblib import Parallel, delayed
from logging.handlers import QueueHandler, QueueListener


# Global queue (will be set in main)
log_queue = None

# Global client
client = OPhysClient("/home/naomi/Desktop/data/V1dd_nwbs")


def listener_config(log_file):
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    return fh


def setup_worker_logger():
    global log_queue
    # Suppress warnings that pop up during train_test_split (mostly because of NS Set 1 -- some images appear 2, 3, 4 times etc)
    warnings.filterwarnings("ignore", message="The least populated class in y has only")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    handler = QueueHandler(log_queue)
    root.addHandler(handler)


def setup_main_logger(log_file, log_queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = QueueHandler(log_queue)
    logger.addHandler(handler)


def depth_volume_mapping_oneplane(volume_id, plane):
    depth_values = {
        1: [50, 66, 82, 98, 114, 130],
        2: [146, 162, 178, 194, 210, 226],
        3: [242, 258, 274, 290, 306, 322],
        4: [338, 354, 370, 386, 402, 418],
        5: [434, 450, 466, 482, 498, 514],
        6: [500],
        7: [525],
        8: [550],
        9: [575],
        "a": [600],
        "b": [625],
        "c": [650],
        "d": [675],
        "e": [700],
        "f": [725],
    }

    return depth_values[volume_id][plane - 1]


def depth_volume_mapping_multiplanes(a, b):
    depth_values = {
        1: {1: 66, 4: 114},
        2: {1: 162, 4: 210},
        3: {1: 258, 4: 306},
        4: {1: 354, 4: 402},
        5: {1: 450, 4: 498},
    }

    return depth_values[a][b[1]]


def get_mouse_name(mouse_id):

    # Set up mouse id mapping to index metrics dataframe correctly
    mouse_id_mapping = {
        427836: "slc5",
        438833: "slc4",
        416296: "teto1",
        409828: "slc2",
    }

    return mouse_id_mapping[mouse_id]


def get_session_id(session, plane=None):
    if plane:
        return (
            "M"
            + str(session.get_mouse_id())
            + "_"
            + str(session.get_column_id())
            + str(session.get_volume_id())
            + "_"
            + str(plane)
        )
    else:
        return (
            "M"
            + str(session.get_mouse_id())
            + "_"
            + str(session.get_column_id())
            + str(session.get_volume_id())
        )


def select_rois(session, plane, metrics_df=None, unduplicated=False):
    rois = session.get_rois(plane)
    if unduplicated:
        if metrics_df is not None:
            rois = metrics_df[
                (metrics_df["mouse_id"] == get_mouse_name(session.get_mouse_id()))
                & (metrics_df["column"] == session.get_column_id())
                & (metrics_df["volume"] == str(session.get_volume_id()))
                & (
                    metrics_df["plane"] == (plane - 1)
                )  # correct for 0-indexing in metrics_df
            ].cell_index.values
            return list(rois)
        else:
            print("Cannot find any unduplicated ROIs...where is metrics_df")
    else:
        return list(
            compress(
                rois,
                session.is_roi_valid(plane=plane),
            )
        )


def get_mean_dff_traces(
    session,
    stimulus_type,
    plane,
    folder_name="/home/naomi/Desktop/data",
):
    # Check if mean dff traces are saved locally
    if "full" in stimulus_type:
        folder = "V1dd_calc_mean_responses_fullfieldDG"
    elif "windowed" in stimulus_type:
        folder = "V1dd_calc_mean_responses_windowedDG"
    elif "natural_images_12" in stimulus_type:
        folder = "V1dd_calc_mean_responses_NS12"
    elif "natural_images" in stimulus_type:
        folder = "V1dd_calc_mean_responses_NS118"
    elif "spont" in stimulus_type:
        folder = "V1dd_calc_mean_responses_spont"
    elif "movie" in stimulus_type:
        folder = "V1dd_calc_mean_responses_natmovie"
    elif "noise" in stimulus_type:
        folder = "V1dd_calc_mean_responses_lsn"
    else:
        print("ERROR, FOLDER DOESN'T EXIST")

    session_id = get_session_id(session=session, plane=plane)

    # Check if the mean traces df is already calculated and saved locally
    if os.path.isfile(
        os.path.join(
            folder_name,
            folder,
            f"{session_id}_mean_response_df.pkl",
        )
    ):
        mean_response_df = pd.read_pickle(
            os.path.join(
                folder_name,
                folder,
                f"{session_id}_mean_response_df.pkl",
            )
        )
        return mean_response_df

    # Load in DFOF traces
    dff_traces = session.get_traces(plane=plane, trace_type="dff")

    # Load in stimulus table and drop any rows with NA values
    stimulus_table, _ = session.get_stimulus_table(stimulus_type)
    stimulus_table = stimulus_table.dropna()
    stimulus_table = stimulus_table[
        stimulus_table["end"]
        < session.get_traces(plane=plane, trace_type="dff").time.values[-1]
    ]

    # Grab the start times and end times of each stimulus
    stim_starts = stimulus_table.start.values
    stim_ends = stimulus_table.end.values

    # Iterate through each pair of start / end and calculate mean DFOF value during stimulus for each ROI
    mean_dff_traces = []
    all_rois = select_rois(session, plane, metrics_df=None, unduplicated=False)
    for roi in all_rois:
        roi_mean_dff_trace = []
        roi_dff_trace = dff_traces[roi]
        for start, end in zip(stim_starts, stim_ends):
            start_idx = np.where(roi_dff_trace.time < start)[0][-1]
            end_idx = np.where(roi_dff_trace.time > end)[0][0]
            roi_mean_dff_trace.append(np.mean(roi_dff_trace[start_idx:end_idx].values))
        mean_dff_traces.append(roi_mean_dff_trace)

    # Convert mean traces list into a dataframe and set column name as ROI index
    mean_dff_traces_df = pd.DataFrame(data=mean_dff_traces).T
    mean_dff_traces_df.columns = all_rois

    # Save mean traces df locally
    pd.to_pickle(
        mean_dff_traces_df,
        os.path.join(
            folder_name,
            folder,
            f"{session_id}_mean_response_df.pkl",
        ),
    )
    return mean_dff_traces_df


def knn_cross_validation(max_neighbors, metric, X_data, Y_data):
    cross_val_scores = []
    possible_neighbors = np.arange(2, max_neighbors + 1, 1)
    for k_neighbors in possible_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric=metric)
        score = cross_val_score(knn, X_data, Y_data, cv=5)
        cross_val_scores.append(np.mean(score))
    best_k = possible_neighbors[np.argmax(cross_val_scores)]
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    return knn, best_k


def get_X_data(
    session,
    plane,
    stimulus_type,
    metrics_df=None,
    unduplicated=False,
    folder_name="/home/naomi/Desktop/data",
):

    X_data_df = pd.DataFrame()
    rois = select_rois(session, plane, metrics_df, unduplicated)
    if rois is not None:
        mean_dff_traces_df = get_mean_dff_traces(
            session=session,
            stimulus_type=stimulus_type,
            plane=plane,
            folder_name=folder_name,
        )
        if mean_dff_traces_df is not None:
            X_data_df = pd.concat([X_data_df, mean_dff_traces_df[rois]], axis=1)
        else:
            print(
                f"Mean dff traces df doesn't exist for {get_session_id(session=session, plane=plane)}"
            )
    else:
        print(
            f"Rois (unduplicated={unduplicated}) don't exist for {get_session_id(session=session, plane=plane)}"
        )
    return X_data_df.values


def get_Y_data(session, plane, stimulus_type, decode_dim):

    stimulus_table, _ = session.get_stimulus_table(stimulus_type)
    stimulus_table = stimulus_table.dropna()
    stimulus_table = stimulus_table[
        stimulus_table["end"]
        < session.get_traces(plane=plane, trace_type="dff").time.values[-1]
    ]

    if "drifting_gratings" in stimulus_type:
        if ("direction" in decode_dim) & ("spatial" not in decode_dim):
            return stimulus_table.direction.values
        if ("spatial" in decode_dim) & ("direction" not in decode_dim):
            return stimulus_table.spatial_frequency.values
        if ("direction" in decode_dim) & ("spatial" in decode_dim):
            temp_Y_data = stimulus_table[["direction", "spatial_frequency"]].values
            Y_data_mapped = []
            for val in temp_Y_data:
                if str(val[1]) == "0.04":
                    if str(val[0]) == "0.0":
                        Y_data_mapped.append(1)
                    elif str(val[0]) == "30.0":
                        Y_data_mapped.append(2)
                    elif str(val[0]) == "60.0":
                        Y_data_mapped.append(3)
                    elif str(val[0]) == "90.0":
                        Y_data_mapped.append(4)
                    elif str(val[0]) == "120.0":
                        Y_data_mapped.append(5)
                    elif str(val[0]) == "150.0":
                        Y_data_mapped.append(6)
                    elif str(val[0]) == "180.0":
                        Y_data_mapped.append(7)
                    elif str(val[0]) == "210.0":
                        Y_data_mapped.append(8)
                    elif str(val[0]) == "240.0":
                        Y_data_mapped.append(9)
                    elif str(val[0]) == "270.0":
                        Y_data_mapped.append(10)
                    elif str(val[0]) == "300.0":
                        Y_data_mapped.append(11)
                    elif str(val[0]) == "330.0":
                        Y_data_mapped.append(12)
                    else:
                        print("ERROR WITH VALUE, 0.04")
                elif str(val[1]) == "0.08":
                    if str(val[0]) == "0.0":
                        Y_data_mapped.append(13)
                    elif str(val[0]) == "30.0":
                        Y_data_mapped.append(14)
                    elif str(val[0]) == "60.0":
                        Y_data_mapped.append(15)
                    elif str(val[0]) == "90.0":
                        Y_data_mapped.append(16)
                    elif str(val[0]) == "120.0":
                        Y_data_mapped.append(17)
                    elif str(val[0]) == "150.0":
                        Y_data_mapped.append(18)
                    elif str(val[0]) == "180.0":
                        Y_data_mapped.append(19)
                    elif str(val[0]) == "210.0":
                        Y_data_mapped.append(20)
                    elif str(val[0]) == "240.0":
                        Y_data_mapped.append(21)
                    elif str(val[0]) == "270.0":
                        Y_data_mapped.append(22)
                    elif str(val[0]) == "300.0":
                        Y_data_mapped.append(23)
                    elif str(val[0]) == "330.0":
                        Y_data_mapped.append(24)
                    else:
                        print("ERROR WITH VALUE, 0.08")
                else:
                    print("ERROR WITH VALUE")
            return Y_data_mapped
    elif "natural_images" in stimulus_type:
        return stimulus_table.image_index.values
    else:
        return None


def run_decoding_one_plane(
    session_id,
    plane,
    stimulus_type,
    repetitions,
    decode_dim,
    folds=5,
    bootstrap=True,
    bootstrap_size=250,
    metrics_df=None,
    unduplicated=False,
    folder_name="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
):

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
        print(f"Performing decoding for session {session_id}")
    except ValueError as e:
        print(f"Error loading session {session_id}: {e}")
        return None

    # Check if the session has the passed in plane
    if plane not in session.get_planes():
        print(f"Plane {plane} not found in session {session_id}. Skipping decoding.")
        return None

    # Figure out if we need to choose "unduplicated" ROIs -- note all ROIs in column 1 are set to "duplicated", but only the 2p data is actually duplicated
    # unduplicated = True --> use unduplicated ROIs for decoding (e.g. 2p data, not from column 1)
    # unduplicated = False --> use duplicated ROIs for decoding (e.g. 3p data)
    column = session.get_column_id()
    num_planes = len(session.get_planes())
    if column != 1:
        unduplicated = True
    elif column == 1 and num_planes > 1:
        print(
            "Skipping decoding for 2p sessions in column 1 -- these have all duplicated ROIs"
        )
        return (
            None  # skip all 2p sessions in column 1 -- these have all duplicated ROIs
        )
    elif column == 1 and num_planes == 1:
        unduplicated = False
    if tag:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{tag}_{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )
    else:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )

    file_name = f"{get_session_id(session)}_plane{plane}.pkl"

    # Check if the decoding results df is already calculated and saved locally
    if os.path.isfile(
        os.path.join(
            decoding_folder_name,
            file_name,
        )
    ):
        indiv_results_df = pd.read_pickle(
            os.path.join(
                decoding_folder_name,
                file_name,
            )
        )
        return indiv_results_df

    mouse_ids = []
    column_ids = []
    volume_ids = []
    plane_group = []
    repetition_nums = []
    test_accuracies = []
    test_accuracies_mean = []
    test_accuracies_std = []
    shuf_test_accuracies = []
    shuf_test_accuracies_mean = []
    shuf_test_accuracies_std = []
    num_k_neighbors = []
    shuf_num_k_neighbors = []

    # Check to see if there is X_data (i.e. if there are neuron responses for this session)
    X_data = get_X_data(
        session, plane, stimulus_type, metrics_df, unduplicated, folder_name
    )
    if X_data is None:
        print("Cannot find any X data...cannot perform decoding")
        return None
    if np.shape(X_data)[1] < bootstrap_size:
        print(
            f"Not enough X_data, only have {np.shape(X_data)[1]} rois, cannot perform decoding"
        )
        return None

    # Check to see if there is Y_data (i.e. if there are corresponding visual stimuli for this session)
    Y_data = get_Y_data(session, plane, stimulus_type, decode_dim)
    if Y_data is None:
        return None

    for i in range(repetitions):
        # print(f"{i}", end=" ")

        # Load in X_data (neural responses)
        X_data = get_X_data(
            session, plane, stimulus_type, metrics_df, unduplicated, folder_name
        )

        if (
            bootstrap
        ):  # select w/ replacement -- to control for different #s of neurons between recs.
            X_data = (
                pd.DataFrame(X_data).T.sample(n=bootstrap_size, replace=True).T.values
            )

        # Load in Y_data (visual stimulus identity)
        Y_data = get_Y_data(session, plane, stimulus_type, decode_dim)

        # Check type of Y_data and make sure it is able to be decoded
        # (i.e. has to all be discrete integer values, not continuous)
        if type(Y_data[0]) is not int:
            lab_enc = preprocessing.LabelEncoder()
            Y_data = lab_enc.fit_transform(Y_data)

        # Perform nested cross-validation
        for shuf_type in [False, True]:
            accuracies, best_ks = (
                [],
                [],
            )  # Store accuracies & best # of neighbors across the folds
            for fold in range(folds):
                x_train, x_test, y_train, y_test = train_test_split(
                    X_data,
                    Y_data,
                    test_size=0.2,
                    random_state=int(i * folds + fold + 1),
                )

                if shuf_type:
                    # Shuffle the training data
                    np.random.shuffle(y_train)

                # Perform grid search for optimal k
                param_grid = {"n_neighbors": list(range(1, 30))}
                knn = KNeighborsClassifier(metric="correlation")
                grid_search = GridSearchCV(
                    knn, param_grid, cv=folds, scoring="accuracy"
                )
                grid_search.fit(x_train, y_train)
                best_k = grid_search.best_params_["n_neighbors"]

                # Fit the model with the best k
                knn = KNeighborsClassifier(n_neighbors=best_k, metric="correlation")
                knn.fit(x_train, y_train)

                # Predict on training and test sets
                # y_pred_train = knn.predict(x_train)
                y_pred_test = knn.predict(x_test)

                # Calculate accuracy scores
                # train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
                test_accuracy = metrics.accuracy_score(y_test, y_pred_test)

                # Store accuracies and best k
                accuracies.append(test_accuracy)
                best_ks.append(best_k)

            # Save cross-val scores, mean of scores, and std of scores
            if shuf_type:
                shuf_test_accuracies.append(accuracies)
                shuf_test_accuracies_mean.append(np.mean(accuracies))
                shuf_test_accuracies_std.append(np.std(accuracies))
                shuf_num_k_neighbors.append(best_ks)
            else:
                test_accuracies.append(accuracies)
                test_accuracies_mean.append(np.mean(accuracies))
                test_accuracies_std.append(np.std(accuracies))
                num_k_neighbors.append(best_ks)

        # Save info about recording + decoding #
        mouse_ids.append(session.get_mouse_id())
        column_ids.append(session.get_column_id())
        volume_ids.append(session.get_volume_id())
        plane_group.append(plane)
        repetition_nums.append(i)

    # Save results in dataframe
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                mouse_ids,
                column_ids,
                volume_ids,
                plane_group,
                repetition_nums,
                test_accuracies,
                test_accuracies_mean,
                test_accuracies_std,
                shuf_test_accuracies,
                shuf_test_accuracies_mean,
                shuf_test_accuracies_std,
                num_k_neighbors,
                shuf_num_k_neighbors,
            )
        ),
        columns=[
            "mouse_id",
            "column_id",
            "volume_id",
            "plane",
            "repetition_num",
            "test_accuracies",
            "test_accuracies_mean",
            "test_accuracies_std",
            "shuf_test_accuracies",
            "shuf_test_accuracies_mean",
            "shuf_test_accuracies_std",
            "num_k_neighbors",
            "shuf_num_k_neighbors",
        ],
    )

    if save_decoding:
        if not os.path.exists(decoding_folder_name):
            os.makedirs(decoding_folder_name)
        pd.to_pickle(
            indiv_results_df,
            os.path.join(decoding_folder_name, file_name),
        )

    return indiv_results_df


def run_decoding_across_planes(
    session_id,
    planes,
    stimulus_type,
    repetitions,
    decode_dim,
    folds=5,
    bootstrap=True,
    bootstrap_size=250,
    metrics_df=None,
    folder_name="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
    log=True,
):
    if log_queue is not None:
        setup_worker_logger()

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
        logging.info(f"{session_id}, {planes}: Loaded") if log else None
    except ValueError as e:
        logging.error(f"{session_id}: {e}: Error loading") if log else None
        return None

    # Check that the correct number of planes is provided
    if len(planes) != 3:
        logging.error("Please provide exactly 3 planes for decoding.") if log else None
        raise ValueError("Please provide exactly 3 planes for decoding.")

    if len(session.get_planes()) < 3:
        (
            logging.info(
                f"{session_id}, {planes}: Not have enough planes for decoding."
            )
            if log
            else None
        )
        return None

    # Figure out if we need to choose "unduplicated" ROIs -- note all ROIs in column 1 are set to "duplicated", but only the 2p data is actually duplicated
    # unduplicated = True --> use unduplicated ROIs for decoding (e.g. 2p data, not from column 1)
    # unduplicated = False --> use duplicated ROIs for decoding (e.g. 3p data)
    column = session.get_column_id()
    num_planes = len(session.get_planes())
    if column != 1:
        unduplicated = True
    elif column == 1 and num_planes > 1:
        (
            logging.info(
                f"{session_id}, {planes}: 2p session in column 1, no unduplicated ROIs."
            )
            if log
            else None
        )
        return (
            None  # skip all 2p sessions in column 1 -- these have all duplicated ROIs
        )
    elif column == 1 and num_planes == 1:
        unduplicated = False

    if tag:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{tag}_{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )
    else:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )

    file_name = f"{get_session_id(session)}_planes{planes[0]}{planes[1]}{planes[2]}.pkl"

    # Check if the decoding results df is already calculated and saved locally
    if os.path.isfile(
        os.path.join(
            decoding_folder_name,
            file_name,
        )
    ):
        indiv_results_df = pd.read_pickle(
            os.path.join(
                decoding_folder_name,
                file_name,
            )
        )
        return indiv_results_df

    mouse_ids = []
    column_ids = []
    volume_ids = []
    plane_group = []
    repetition_nums = []
    test_accuracies = []
    test_accuracies_mean = []
    test_accuracies_std = []
    shuf_test_accuracies = []
    shuf_test_accuracies_mean = []
    shuf_test_accuracies_std = []
    num_k_neighbors = []
    shuf_num_k_neighbors = []

    # Load X_data (i.e. if there are neuron responses for this session)
    X_data = np.array([])  # Initialize X_data as an empty array
    for plane in planes:
        if len(X_data) == 0:
            # If X_data is empty, initialize it with the first plane's data
            X_data = get_X_data(
                session=session,
                plane=plane,
                stimulus_type=stimulus_type,
                metrics_df=metrics_df,
                unduplicated=unduplicated,
                folder_name=folder_name,
            )
        else:
            # Concatenate the new plane's data to the existing X_data
            X_data = np.concatenate(
                (
                    X_data,
                    get_X_data(
                        session=session,
                        plane=plane,
                        stimulus_type=stimulus_type,
                        metrics_df=metrics_df,
                        unduplicated=unduplicated,
                        folder_name=folder_name,
                    ),
                ),
                axis=1,
            )

    # Check if there is enough X_data for decoding
    if X_data is None:
        (
            logging.info(
                f"{session_id}, {planes}: Cannot find any X data...cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if np.shape(X_data)[1] < bootstrap_size:
        (
            logging.info(
                f"{session_id}, {planes}: Not enough X_data, only have {np.shape(X_data)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None

    # Load and check to see if there is Y_data (i.e. if there are corresponding visual stimuli for this session)
    ## can just load it once for all planes, since they are from the same session
    Y_data = get_Y_data(session, planes[0], stimulus_type, decode_dim)
    if Y_data is None:
        return None

    for i in range(repetitions):
        if (i + 1) % int(repetitions / 20) == 0 or i == repetitions - 1:
            # Log progress every 5% of repetitions or at the last repetition
            (
                logging.info(f"{session_id}, {planes}: repetition {i+1}/{repetitions}")
                if log
                else None
            )
        if (
            bootstrap
        ):  # select w/ replacement -- to control for different #s of neurons between recs.
            X_data_copy = (
                pd.DataFrame(X_data).T.sample(n=bootstrap_size, replace=True).T.values
            )
        else:
            X_data_copy = X_data.copy()

        Y_data_copy = Y_data.copy()

        # Check type of Y_data and make sure it is able to be decoded
        # (i.e. has to all be discrete integer values, not continuous)
        if type(Y_data_copy[0]) is not int:
            lab_enc = preprocessing.LabelEncoder()
            Y_data_copy = lab_enc.fit_transform(Y_data_copy)

        # Perform nested cross-validation
        for shuf_type in [False, True]:
            accuracies, best_ks = (
                [],
                [],
            )  # Store accuracies & best # of neighbors across the folds
            for fold in range(folds):
                x_train, x_test, y_train, y_test = train_test_split(
                    X_data_copy,
                    Y_data_copy,
                    test_size=0.2,
                    random_state=int(i * folds + fold + 1),
                )

                if shuf_type:
                    # Shuffle the training data
                    np.random.shuffle(y_train)

                # Perform grid search for optimal k
                param_grid = {"n_neighbors": list(range(1, 30))}
                knn = KNeighborsClassifier(metric="correlation")
                grid_search = GridSearchCV(
                    knn, param_grid, cv=folds, scoring="accuracy"
                )
                grid_search.fit(x_train, y_train)
                best_k = grid_search.best_params_["n_neighbors"]

                # Fit the model with the best k
                knn = KNeighborsClassifier(n_neighbors=best_k, metric="correlation")
                knn.fit(x_train, y_train)

                # Predict on training and test sets
                # y_pred_train = knn.predict(x_train)
                y_pred_test = knn.predict(x_test)

                # Calculate accuracy scores
                # train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
                test_accuracy = metrics.accuracy_score(y_test, y_pred_test)

                # Store accuracies and best k
                accuracies.append(test_accuracy)
                best_ks.append(best_k)

            # Save cross-val scores, mean of scores, and std of scores
            if shuf_type:
                shuf_test_accuracies.append(accuracies)
                shuf_test_accuracies_mean.append(np.mean(accuracies))
                shuf_test_accuracies_std.append(np.std(accuracies))
                shuf_num_k_neighbors.append(best_ks)
            else:
                test_accuracies.append(accuracies)
                test_accuracies_mean.append(np.mean(accuracies))
                test_accuracies_std.append(np.std(accuracies))
                num_k_neighbors.append(best_ks)

        # Save info about recording + decoding #
        mouse_ids.append(session.get_mouse_id())
        column_ids.append(session.get_column_id())
        volume_ids.append(session.get_volume_id())
        plane_group.append(planes)
        repetition_nums.append(i)

    # Save results in dataframe
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                mouse_ids,
                column_ids,
                volume_ids,
                plane_group,
                repetition_nums,
                test_accuracies,
                test_accuracies_mean,
                test_accuracies_std,
                shuf_test_accuracies,
                shuf_test_accuracies_mean,
                shuf_test_accuracies_std,
                num_k_neighbors,
                shuf_num_k_neighbors,
            )
        ),
        columns=[
            "mouse_id",
            "column_id",
            "volume_id",
            "planes",
            "repetition_num",
            "test_accuracies",
            "test_accuracies_mean",
            "test_accuracies_std",
            "shuf_test_accuracies",
            "shuf_test_accuracies_mean",
            "shuf_test_accuracies_std",
            "num_k_neighbors",
            "shuf_num_k_neighbors",
        ],
    )

    if save_decoding:
        if not os.path.exists(decoding_folder_name):
            os.makedirs(decoding_folder_name)
        pd.to_pickle(
            indiv_results_df,
            os.path.join(decoding_folder_name, file_name),
        )
    (
        logging.info(f"{session_id}, {planes}: Decoding completed and saved")
        if log
        else None
    )

    return indiv_results_df


def run_decoding_multistim_one_plane(
    session_id,
    plane,
    stimulus_type_training,
    stimulus_type_testing,
    repetitions,
    decode_dim,
    folds=5,
    bootstrap=False,
    bootstrap_size=1,
    metrics_df=None,
    folder_name="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
    log=True,
):
    if log_queue is not None:
        setup_worker_logger()

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
        logging.info(f"{session_id}: Loaded") if log else None
    except ValueError as e:
        logging.error(f"{session_id}: {e}: Error loading") if log else None
        return None

    # Check if the session has the passed in plane
    if plane not in session.get_planes():
        logging.info(f"{session_id}: Plane {plane} not found") if log else None
        return None

    # Figure out if we need to choose "unduplicated" ROIs -- note all ROIs in column 1 are set to "duplicated", but only the 2p data is actually duplicated
    # unduplicated = True --> use unduplicated ROIs for decoding (e.g. 2p data, not from column 1)
    # unduplicated = False --> use duplicated ROIs for decoding (e.g. 3p data)
    column = session.get_column_id()
    num_planes = len(session.get_planes())
    if column != 1:
        unduplicated = True
    elif column == 1 and num_planes > 1:
        (
            logging.info(
                f"{session_id}, {plane}: 2p session in column 1, no unduplicated ROIs."
            )
            if log
            else None
        )
        return (
            None  # skip all 2p sessions in column 1 -- these have all duplicated ROIs
        )
    elif column == 1 and num_planes == 1:
        unduplicated = False

    if tag:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{tag}_TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_Boot{bootstrap_size}_Rep{repetitions}",
        )
    else:
        decoding_folder_name = os.path.join(
            results_folder,
            f"TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )

    file_name = f"{get_session_id(session)}_plane{plane}.pkl"

    # Check if the decoding results df is already calculated and saved locally
    if os.path.isfile(
        os.path.join(
            decoding_folder_name,
            file_name,
        )
    ):
        indiv_results_df = pd.read_pickle(
            os.path.join(
                decoding_folder_name,
                file_name,
            )
        )
        return indiv_results_df

    mouse_ids = []
    column_ids = []
    volume_ids = []
    planes = []
    repetition_nums = []
    val_accuracies = []
    val_accuracies_mean = []
    val_accuracies_std = []
    test_accuracies = []
    test_accuracies_mean = []
    test_accuracies_std = []
    shuf_val_accuracies = []
    shuf_val_accuracies_mean = []
    shuf_val_accuracies_std = []
    shuf_test_accuracies = []
    shuf_test_accuracies_mean = []
    shuf_test_accuracies_std = []
    num_k_neighbors = []
    shuf_num_k_neighbors = []

    # Load X_data (i.e. if there are neuron responses for this session)
    X_data_training = get_X_data(
        session=session,
        plane=plane,
        stimulus_type=stimulus_type_training,
        metrics_df=metrics_df,
        unduplicated=unduplicated,
        folder_name=folder_name,
    )
    X_data_testing = get_X_data(
        session=session,
        plane=plane,
        stimulus_type=stimulus_type_testing,
        metrics_df=metrics_df,
        unduplicated=unduplicated,
        folder_name=folder_name,
    )

    # Check if there is enough X_data for decoding
    if X_data_training is None:
        (
            logging.info(
                f"{session_id}, {plane}: Cannot find any X data (training)...cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if np.shape(X_data_training)[1] < bootstrap_size:
        (
            logging.info(
                f"{session_id}, {plane}: Not enough X_data (training), only have {np.shape(X_data_training)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if X_data_testing is None:
        (
            logging.info(
                f"{session_id}, {plane}: Cannot find any X data (testing)...cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if np.shape(X_data_testing)[1] < bootstrap_size:
        (
            logging.info(
                f"{session_id}, {plane}: Not enough X_data (testing), only have {np.shape(X_data_testing)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None

    # Load and check to see if there is Y_data (i.e. if there are corresponding visual stimuli for this session)
    ## can just load it once for all planes, since they are from the same session
    Y_data_training = get_Y_data(session, plane, stimulus_type_training, decode_dim)
    # Check type of Y_data and make sure it is able to be decoded
    # (i.e. has to all be discrete integer values, not continuous)
    if type(Y_data_training[0]) is not int:
        lab_enc = preprocessing.LabelEncoder()
        Y_data_training = lab_enc.fit_transform(Y_data_training)
    if Y_data_training is None:
        return None

    # Load Y_data for testing (i.e. if there are corresponding visual stimuli for this session)
    Y_data_testing = get_Y_data(session, plane, stimulus_type_testing, decode_dim)
    if type(Y_data_testing[0]) is not int:
        lab_enc = preprocessing.LabelEncoder()
        Y_data_testing = lab_enc.fit_transform(Y_data_testing)
    if Y_data_testing is None:
        return None

    for i in range(repetitions):
        if repetitions > 50:
            if (i + 1) % int(repetitions / 20) == 0 or i == repetitions - 1:
                # Log progress every 5% of repetitions or at the last repetition
                (
                    logging.info(
                        f"{session_id}, {plane}: repetition {i+1}/{repetitions}"
                    )
                    if log
                    else None
                )
        if (
            bootstrap
        ):  # select w/ replacement -- to control for different #s of neurons between recs.
            X_data_training_copy = (
                pd.DataFrame(X_data_training)
                .T.sample(n=bootstrap_size, replace=True)
                .T.values
            )
            X_data_testing_copy = (
                pd.DataFrame(X_data_testing)
                .T.sample(n=bootstrap_size, replace=True)
                .T.values
            )
        else:
            X_data_training_copy = X_data_training.copy()
            X_data_testing_copy = X_data_testing.copy()

        Y_data_training_copy = Y_data_training.copy()
        Y_data_testing_copy = Y_data_testing.copy()

        # Perform nested cross-validation
        for shuf_type in [False, True]:
            val_accuracies_folds, test_accuracies_folds, best_ks = (
                [],
                [],
                [],
            )  # Store accuracies & best # of neighbors across the folds
            for fold in range(folds):
                x_train, x_val, y_train, y_val = train_test_split(
                    X_data_training_copy,
                    Y_data_training_copy,
                    test_size=0.2,
                    random_state=int(i * folds + fold + 1),
                )

                if shuf_type:
                    # Shuffle the training data
                    np.random.shuffle(y_train)

                # Perform grid search for optimal k
                param_grid = {"n_neighbors": list(range(1, 30))}
                knn = KNeighborsClassifier(metric="correlation")
                grid_search = GridSearchCV(
                    knn, param_grid, cv=folds, scoring="accuracy"
                )
                grid_search.fit(x_train, y_train)
                best_k = grid_search.best_params_["n_neighbors"]

                # Fit the model with the best k
                knn = KNeighborsClassifier(n_neighbors=best_k, metric="correlation")
                knn.fit(x_train, y_train)

                # Predict on training, val, and test sets
                # y_pred_train = knn.predict(x_train)
                y_pred_val = knn.predict(x_val)
                y_pred_test = knn.predict(X_data_testing_copy)

                # Calculate accuracy scores
                # train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
                val_accuracy = metrics.accuracy_score(y_val, y_pred_val)
                test_accuracy = metrics.accuracy_score(Y_data_testing_copy, y_pred_test)

                # Store validation set accuracies and best k
                val_accuracies_folds.append(val_accuracy)
                test_accuracies_folds.append(test_accuracy)
                best_ks.append(best_k)

            # Save cross-val scores, mean of scores, and std of scores
            if shuf_type:
                shuf_val_accuracies.append(val_accuracies_folds)
                shuf_val_accuracies_mean.append(np.mean(val_accuracies_folds))
                shuf_val_accuracies_std.append(np.std(val_accuracies_folds))
                shuf_test_accuracies.append(test_accuracies_folds)
                shuf_test_accuracies_mean.append(np.mean(test_accuracies_folds))
                shuf_test_accuracies_std.append(np.std(test_accuracies_folds))
                shuf_num_k_neighbors.append(best_ks)
            else:
                val_accuracies.append(val_accuracies_folds)
                val_accuracies_mean.append(np.mean(val_accuracies_folds))
                val_accuracies_std.append(np.std(val_accuracies_folds))
                test_accuracies.append(test_accuracies_folds)
                test_accuracies_mean.append(np.mean(test_accuracies_folds))
                test_accuracies_std.append(np.std(test_accuracies_folds))
                num_k_neighbors.append(best_ks)

        # Save info about recording + decoding #
        mouse_ids.append(session.get_mouse_id())
        column_ids.append(session.get_column_id())
        volume_ids.append(session.get_volume_id())
        planes.append(plane)
        repetition_nums.append(i)

    # Save results in dataframe
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                mouse_ids,
                column_ids,
                volume_ids,
                planes,
                repetition_nums,
                val_accuracies,
                val_accuracies_mean,
                val_accuracies_std,
                test_accuracies,
                test_accuracies_mean,
                test_accuracies_std,
                shuf_val_accuracies,
                shuf_val_accuracies_mean,
                shuf_val_accuracies_std,
                shuf_test_accuracies,
                shuf_test_accuracies_mean,
                shuf_test_accuracies_std,
                num_k_neighbors,
                shuf_num_k_neighbors,
            )
        ),
        columns=[
            "mouse_id",
            "column_id",
            "volume_id",
            "plane",
            "repetition_num",
            "val_accuracies",
            "val_accuracies_mean",
            "val_accuracies_std",
            "test_accuracies",
            "test_accuracies_mean",
            "test_accuracies_std",
            "shuf_val_accuracies",
            "shuf_val_accuracies_mean",
            "shuf_val_accuracies_std",
            "shuf_test_accuracies",
            "shuf_test_accuracies_mean",
            "shuf_test_accuracies_std",
            "num_k_neighbors",
            "shuf_num_k_neighbors",
        ],
    )

    if save_decoding:
        if not os.path.exists(decoding_folder_name):
            os.makedirs(decoding_folder_name)
        pd.to_pickle(
            indiv_results_df,
            os.path.join(decoding_folder_name, file_name),
        )
    (
        logging.info(f"{session_id}, {plane}: Decoding completed and saved")
        if log
        else None
    )

    return indiv_results_df


def run_decoding_multistim_across_planes(
    session_id,
    planes,
    stimulus_type_training,
    stimulus_type_testing,
    repetitions,
    decode_dim,
    folds=5,
    bootstrap=True,
    bootstrap_size=250,
    metrics_df=None,
    folder_name="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
    log=True,
    chunk_range=None,
):
    if log_queue is not None:
        setup_worker_logger()

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
        # logging.info(f"{session_id}, {planes}: Loaded") if log else None
    except ValueError as e:
        logging.error(f"{session_id}: {e}: Error loading") if log else None
        return None

    # Check that the correct number of planes is provided
    if len(planes) != 3:
        logging.error("Please provide exactly 3 planes for decoding.") if log else None
        raise ValueError("Please provide exactly 3 planes for decoding.")

    if len(session.get_planes()) < 3:
        (
            logging.info(
                f"{session_id}, {planes}: Not have enough planes for decoding."
            )
            if log
            else None
        )
        return None

    # Figure out if we need to choose "unduplicated" ROIs -- note all ROIs in column 1 are set to "duplicated", but only the 2p data is actually duplicated
    # unduplicated = True --> use unduplicated ROIs for decoding (e.g. 2p data, not from column 1)
    # unduplicated = False --> use duplicated ROIs for decoding (e.g. 3p data)
    column = session.get_column_id()
    num_planes = len(session.get_planes())
    if column != 1:
        unduplicated = True
    elif column == 1 and num_planes > 1:
        # (
        #     logging.info(
        #         f"{session_id}, {planes}: 2p session in column 1, no unduplicated ROIs."
        #     )
        #     if log
        #     else None
        # )
        return (
            None  # skip all 2p sessions in column 1 -- these have all duplicated ROIs
        )
    elif column == 1 and num_planes == 1:
        unduplicated = False

    if tag:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{tag}_TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_Boot{bootstrap_size}_Rep{repetitions}",
        )
    else:
        decoding_folder_name = os.path.join(
            results_folder,
            f"TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}",
        )

    if chunk_range is None:
        file_name = f"{get_session_id(session)}_planes{planes[0]}{planes[1]}{planes[2]}_reps0-{repetitions}.pkl"
    else:
        file_name = f"{get_session_id(session)}_planes{planes[0]}{planes[1]}{planes[2]}_reps{chunk_range[0]}-{chunk_range[1]}.pkl"

    # Check if the decoding results df is already calculated and saved locally
    if os.path.isfile(
        os.path.join(
            decoding_folder_name,
            file_name,
        )
    ):
        indiv_results_df = pd.read_pickle(
            os.path.join(
                decoding_folder_name,
                file_name,
            )
        )
        return indiv_results_df

    mouse_ids = []
    column_ids = []
    volume_ids = []
    plane_group = []
    repetition_nums = []
    val_accuracies = []
    val_accuracies_mean = []
    val_accuracies_std = []
    test_accuracies = []
    test_accuracies_mean = []
    test_accuracies_std = []
    shuf_val_accuracies = []
    shuf_val_accuracies_mean = []
    shuf_val_accuracies_std = []
    shuf_test_accuracies = []
    shuf_test_accuracies_mean = []
    shuf_test_accuracies_std = []
    num_k_neighbors = []
    shuf_num_k_neighbors = []

    # Load X_data (i.e. if there are neuron responses for this session)
    # X_data_training = get_X_data(
    #     session=session,
    #     plane=planes[0],
    #     stimulus_type=stimulus_type_training,
    #     metrics_df=metrics_df,
    #     unduplicated=unduplicated,
    #     folder_name=folder_name,
    # )
    # X_data_testing = get_X_data(
    #     session=session,
    #     plane=planes[0],
    #     stimulus_type=stimulus_type_testing,
    #     metrics_df=metrics_df,
    #     unduplicated=unduplicated,
    #     folder_name=folder_name,
    # )

    X_data_training, X_data_testing = pd.DataFrame(), pd.DataFrame()
    for plane in planes:
        if len(X_data_training) == 0:
            # If X_data_training is empty, initialize it with the first plane's data
            X_data_training = get_X_data(
                session=session,
                plane=plane,
                stimulus_type=stimulus_type_training,
                metrics_df=metrics_df,
                unduplicated=unduplicated,
                folder_name=folder_name,
            )
            X_data_testing = get_X_data(
                session=session,
                plane=plane,
                stimulus_type=stimulus_type_testing,
                metrics_df=metrics_df,
                unduplicated=unduplicated,
                folder_name=folder_name,
            )
        else:
            # Concatenate the new plane's data to the existing X_data
            X_data_training = np.concatenate(
                (
                    X_data_training,
                    get_X_data(
                        session=session,
                        plane=plane,
                        stimulus_type=stimulus_type_training,
                        metrics_df=metrics_df,
                        unduplicated=unduplicated,
                        folder_name=folder_name,
                    ),
                ),
                axis=1,
            )
            X_data_testing = np.concatenate(
                (
                    X_data_testing,
                    get_X_data(
                        session=session,
                        plane=plane,
                        stimulus_type=stimulus_type_testing,
                        metrics_df=metrics_df,
                        unduplicated=unduplicated,
                        folder_name=folder_name,
                    ),
                ),
                axis=1,
            )

    # Check if there is enough X_data for decoding
    if X_data_training is None:
        (
            logging.info(
                f"{session_id}, {planes}: Cannot find any X data (training)...cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if np.shape(X_data_training)[1] < bootstrap_size:
        (
            logging.info(
                f"{session_id}, {planes}: Not enough X_data (training), only have {np.shape(X_data_training)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if X_data_testing is None:
        (
            logging.info(
                f"{session_id}, {planes}: Cannot find any X data (testing)...cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if np.shape(X_data_testing)[1] < bootstrap_size:
        (
            logging.info(
                f"{session_id}, {planes}: Not enough X_data (testing), only have {np.shape(X_data_testing)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None

    # Load and check to see if there is Y_data (i.e. if there are corresponding visual stimuli for this session)
    ## can just load it once for all planes, since they are from the same session
    Y_data_training = get_Y_data(session, planes[0], stimulus_type_training, decode_dim)
    # Check type of Y_data and make sure it is able to be decoded
    # (i.e. has to all be discrete integer values, not continuous)
    if type(Y_data_training[0]) is not int:
        lab_enc = preprocessing.LabelEncoder()
        Y_data_training = lab_enc.fit_transform(Y_data_training)
    if Y_data_training is None:
        return None

    # Load Y_data for testing (i.e. if there are corresponding visual stimuli for this session)
    Y_data_testing = get_Y_data(session, planes[0], stimulus_type_testing, decode_dim)
    if type(Y_data_testing[0]) is not int:
        lab_enc = preprocessing.LabelEncoder()
        Y_data_testing = lab_enc.fit_transform(Y_data_testing)
    if Y_data_testing is None:
        return None

    for i in range(chunk_range[0], chunk_range[1]):
        # if (i + 1) % int(repetitions / 20) == 0 or i == repetitions - 1:
        #     # Log progress every 5% of repetitions or at the last repetition
        #     (
        #         logging.info(f"{session_id}, {planes}: repetition {i+1}/{repetitions}")
        #         if log
        #         else None
        #     )
        if (
            bootstrap
        ):  # select w/ replacement -- to control for different #s of neurons between recs.
            cols = np.random.choice(
                X_data_training.shape[1], bootstrap_size, replace=True
            )
            X_data_training_copy = X_data_training[:, cols]
            X_data_testing_copy = X_data_testing[:, cols]
        else:
            X_data_training_copy = X_data_training.copy()
            X_data_testing_copy = X_data_testing.copy()

        Y_data_training_copy = Y_data_training.copy()
        Y_data_testing_copy = Y_data_testing.copy()

        # Perform nested cross-validation
        for shuf_type in [False, True]:
            val_accuracies_folds, test_accuracies_folds, best_ks = (
                [],
                [],
                [],
            )  # Store accuracies & best # of neighbors across the folds
            for fold in range(folds):
                x_train, x_val, y_train, y_val = train_test_split(
                    X_data_training_copy,
                    Y_data_training_copy,
                    test_size=0.2,
                    random_state=int(i * folds + fold + 1),
                )

                if shuf_type:
                    # Shuffle the training data
                    np.random.shuffle(y_train)

                # Perform grid search for optimal k
                param_grid = {"n_neighbors": list(range(1, 30))}
                knn = KNeighborsClassifier(metric="correlation")
                grid_search = GridSearchCV(
                    knn, param_grid, cv=folds, scoring="accuracy"
                )
                grid_search.fit(x_train, y_train)
                best_k = grid_search.best_params_["n_neighbors"]

                # Fit the model with the best k
                knn = KNeighborsClassifier(n_neighbors=best_k, metric="correlation")
                knn.fit(x_train, y_train)

                # Predict on training, val, and test sets
                # y_pred_train = knn.predict(x_train)
                y_pred_val = knn.predict(x_val)
                y_pred_test = knn.predict(X_data_testing_copy)

                # Calculate accuracy scores
                # train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
                val_accuracy = metrics.accuracy_score(y_val, y_pred_val)
                test_accuracy = metrics.accuracy_score(Y_data_testing_copy, y_pred_test)

                # Store validation set accuracies and best k
                val_accuracies_folds.append(val_accuracy)
                test_accuracies_folds.append(test_accuracy)
                best_ks.append(best_k)

                del (
                    knn,
                    grid_search,
                    x_train,
                    x_val,
                    y_train,
                    y_val,
                    val_accuracy,
                    test_accuracy,
                    best_k,
                )
                gc.collect()

            # Save cross-val scores, mean of scores, and std of scores
            if shuf_type:
                shuf_val_accuracies.append(val_accuracies_folds)
                shuf_val_accuracies_mean.append(np.mean(val_accuracies_folds))
                shuf_val_accuracies_std.append(np.std(val_accuracies_folds))
                shuf_test_accuracies.append(test_accuracies_folds)
                shuf_test_accuracies_mean.append(np.mean(test_accuracies_folds))
                shuf_test_accuracies_std.append(np.std(test_accuracies_folds))
                shuf_num_k_neighbors.append(best_ks)
            else:
                val_accuracies.append(val_accuracies_folds)
                val_accuracies_mean.append(np.mean(val_accuracies_folds))
                val_accuracies_std.append(np.std(val_accuracies_folds))
                test_accuracies.append(test_accuracies_folds)
                test_accuracies_mean.append(np.mean(test_accuracies_folds))
                test_accuracies_std.append(np.std(test_accuracies_folds))
                num_k_neighbors.append(best_ks)

        # Save info about recording + decoding #
        mouse_ids.append(session.get_mouse_id())
        column_ids.append(session.get_column_id())
        volume_ids.append(session.get_volume_id())
        plane_group.append(planes)
        repetition_nums.append(i)

        # Clean up memory
        del val_accuracies_folds, test_accuracies_folds, best_ks
        del (
            X_data_training_copy,
            X_data_testing_copy,
            Y_data_training_copy,
            Y_data_testing_copy,
        )
        gc.collect()

    # Save results in dataframe
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                mouse_ids,
                column_ids,
                volume_ids,
                plane_group,
                repetition_nums,
                val_accuracies,
                val_accuracies_mean,
                val_accuracies_std,
                test_accuracies,
                test_accuracies_mean,
                test_accuracies_std,
                shuf_val_accuracies,
                shuf_val_accuracies_mean,
                shuf_val_accuracies_std,
                shuf_test_accuracies,
                shuf_test_accuracies_mean,
                shuf_test_accuracies_std,
                num_k_neighbors,
                shuf_num_k_neighbors,
            )
        ),
        columns=[
            "mouse_id",
            "column_id",
            "volume_id",
            "planes",
            "repetition_num",
            "val_accuracies",
            "val_accuracies_mean",
            "val_accuracies_std",
            "test_accuracies",
            "test_accuracies_mean",
            "test_accuracies_std",
            "shuf_val_accuracies",
            "shuf_val_accuracies_mean",
            "shuf_val_accuracies_std",
            "shuf_test_accuracies",
            "shuf_test_accuracies_mean",
            "shuf_test_accuracies_std",
            "num_k_neighbors",
            "shuf_num_k_neighbors",
        ],
    )

    # Clean up memory

    del mouse_ids, column_ids, volume_ids, plane_group, repetition_nums
    del val_accuracies, val_accuracies_mean, val_accuracies_std
    del test_accuracies, test_accuracies_mean, test_accuracies_std
    del shuf_val_accuracies, shuf_val_accuracies_mean, shuf_val_accuracies_std
    del shuf_test_accuracies, shuf_test_accuracies_mean, shuf_test_accuracies_std
    del num_k_neighbors, shuf_num_k_neighbors
    del session
    gc.collect()

    if save_decoding:
        if not os.path.exists(decoding_folder_name):
            os.makedirs(decoding_folder_name)
        pd.to_pickle(
            indiv_results_df,
            os.path.join(decoding_folder_name, file_name),
        )
    (
        logging.info(
            f"{session_id}, {planes}, {chunk_range}: Decoding completed and saved"
        )
        if log
        else None
    )

    return indiv_results_df


if __name__ == "__main__":
    tag = "2025_0804_3"  # Tag for saving results -- will take form "{tag}_{stim_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}"

    # Set up logging
    log_dir = os.path.join(ARTIFACT_DIR, "decoding_analyses/logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"decoding_{tag}_{timestamp}.log")
    manager = Manager()
    log_queue = manager.Queue()
    file_handler = listener_config(log_file)
    listener = QueueListener(log_queue, file_handler)
    listener.start()

    # Set up main logger to also use the queue
    setup_main_logger(log_file, log_queue)

    ## Decoding parameters
    repetitions = 1000  # Number of bootstrap repetitions
    bootstrap = True  # If True, perform bootstrap sampling; if False, use all ROIs in the session
    bootstrap_size = 250  # Size of bootstrap sample; or threshold for minimum number of ROIs to perform decoding (if bootstrap=False)
    data_folder = "/home/naomi/Desktop/data"
    path_to_nwbs = f"{data_folder}/V1dd_nwbs"
    path_to_metrics = f"{data_folder}/all_metrics_240426.csv"
    results_folder = f"{data_folder}/decoding_results"
    one_plane = False  # If True, decode across a single plane; if False, decode across multiple planes (e.g. just 2p data)
    multi_stim = True  # If True, decode across multiple stimuli (e.g. DGW and DGF); if False, decode a single stimulus type (e.g. just DGW)
    chunk_size = 25  # Size of chunks for parallel processing
    repetition_chunks = [(i, i + chunk_size) for i in range(0, repetitions, chunk_size)]

    ## Load in the client
    client = OPhysClient(path_to_nwbs)

    ## Load in metrics
    metrics_df = pd.read_csv(path_to_metrics)

    ## Stim types and decode dimensions
    stim_types = [
        "drifting_gratings_full",
        "drifting_gratings_windowed",
        "natural_images_12",
        "natural_images",
    ]
    decode_dims = {
        "drifting_gratings_full": "direction",
        "drifting_gratings_windowed": "direction",
        "natural_images_12": "image_index",
        "natural_images": "image_index",
    }
    multi_stim_pairs = {
        "drifting_gratings_full": "drifting_gratings_windowed",
        "drifting_gratings_windowed": "drifting_gratings_full",
        "natural_images_12": "natural_images",
        "natural_images": "natural_images_12",
    }

    logging.info(
        f"Decoding parameters: {repetitions} repetitions, {bootstrap_size} bootstrap size, tag: {tag}, bootstrap, {bootstrap}, one_plane: {one_plane}, multi_stim: {multi_stim}"
    )

    # apply multiprocessing to run decoding in parallel
    if multi_stim:
        if one_plane:
            Parallel(n_jobs=30)(
                delayed(run_decoding_multistim_one_plane)(
                    session_id=session_id,
                    plane=plane,
                    stimulus_type_training=stim_type,
                    stimulus_type_testing=multi_stim_pairs[stim_type],
                    repetitions=repetitions,
                    decode_dim=decode_dims[stim_type],
                    bootstrap=bootstrap,
                    bootstrap_size=bootstrap_size,
                    metrics_df=metrics_df,
                    folder_name=data_folder,
                    save_decoding=True,
                    results_folder=results_folder,
                    tag=tag,
                )
                for session_id in client.get_all_session_ids()
                for plane in [1, 2, 3, 4, 5, 6]
                for stim_type in stim_types
            )
        else:
            Parallel(n_jobs=30)(
                delayed(run_decoding_multistim_across_planes)(
                    session_id=session_id,
                    planes=planes,
                    stimulus_type_training=stim_type,
                    stimulus_type_testing=multi_stim_pairs[stim_type],
                    repetitions=repetitions,
                    decode_dim=decode_dims[stim_type],
                    bootstrap=bootstrap,
                    bootstrap_size=bootstrap_size,
                    metrics_df=metrics_df,
                    folder_name=data_folder,
                    save_decoding=True,
                    results_folder=results_folder,
                    tag=tag,
                    log=True,
                    chunk_range=chunk_range,
                )
                for session_id in client.get_all_session_ids()
                for planes in [[1, 2, 3], [4, 5, 6]]
                for stim_type in stim_types
                for chunk_range in repetition_chunks
            )
    else:
        if one_plane:
            Parallel(n_jobs=30)(
                delayed(run_decoding_one_plane)(
                    session_id=session_id,
                    plane=plane,
                    stimulus_type=stim_type,
                    repetitions=repetitions,
                    decode_dim=decode_dims[stim_type],
                    bootstrap=bootstrap,
                    bootstrap_size=bootstrap_size,
                    metrics_df=metrics_df,
                    folder_name=data_folder,
                    save_decoding=True,
                    results_folder=results_folder,
                    tag=tag,
                )
                for session_id in client.get_all_session_ids()
                for plane in [1, 2, 3, 4, 5, 6]
                for stim_type in stim_types
            )
        else:
            Parallel(n_jobs=30)(
                delayed(run_decoding_across_planes)(
                    session_id=session_id,
                    planes=planes,
                    stimulus_type=stim_type,
                    repetitions=repetitions,
                    decode_dim=decode_dims[stim_type],
                    bootstrap=bootstrap,
                    bootstrap_size=bootstrap_size,
                    metrics_df=metrics_df,
                    folder_name=data_folder,
                    save_decoding=True,
                    results_folder=results_folder,
                    tag=tag,
                    log=True,
                )
                for session_id in client.get_all_session_ids()
                for planes in [[1, 2, 3], [4, 5, 6]]
                for stim_type in stim_types
            )

    # Convert results to DataFrame
    all_results_df = pd.DataFrame()
    for stim_type in stim_types:
        decode_dim = decode_dims[
            stim_type
        ]  # Assuming only one decode dimension per stim type

        if multi_stim:
            path_name = f"/home/naomi/Desktop/data/decoding_results/{tag}_TRAIN{stim_type}_TEST{multi_stim_pairs[stim_type]}_Boot{bootstrap_size}_Rep{repetitions}"
        else:
            path_name = f"/home/naomi/Desktop/data/decoding_results/{tag}_{stim_type}_{decode_dim}_Boot1_Rep1"
        results_df = pd.DataFrame()
        for filename in os.listdir(path_name):
            f = os.path.join(path_name, filename)
            results_df = pd.concat([results_df, pd.read_pickle(f)])

        if multi_stim:
            results_df["train_stim_type"] = stim_type
            results_df["test_stim_type"] = multi_stim_pairs[stim_type]
        else:
            results_df["stim_type"] = stim_type
        results_df["decode_dim"] = decode_dim

        all_results_df = pd.concat([all_results_df, results_df], axis=0)

    if one_plane:
        all_results_df["depth"] = all_results_df.apply(
            lambda x: depth_volume_mapping_oneplane(
                volume_id=x["volume_id"], plane=x["plane"]
            ),
            axis=1,
        )
    else:
        all_results_df["depth"] = all_results_df.apply(
            lambda x: depth_volume_mapping_multiplanes(a=x["volume_id"], b=x["planes"]),
            axis=1,
        )

    all_results_path = os.path.join(
        ARTIFACT_DIR,
        f"decoding_analyses/{tag}_Boot{bootstrap_size}_Reps{repetitions}.pkl",
    )
    all_results_df.to_pickle(all_results_path)
    logging.info(f"Saved all results in a dataframe: {all_results_path}")

    listener.stop()

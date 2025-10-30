import gc
import os
import argparse
import numpy as np
import pandas as pd
from itertools import compress
from v1dd_public import ARTIFACT_DIR
from sklearn import metrics
from allen_v1dd.client import OPhysClient

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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


def setup_main_logger(log_queue):
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


def get_Y_data(session, plane, stimulus_type, decode_dim, time_fractions=None):

    stimulus_table, _ = session.get_stimulus_table(stimulus_type)
    stimulus_table = stimulus_table.dropna()
    stimulus_table = stimulus_table[
        stimulus_table["end"]
        < session.get_traces(plane=plane, trace_type="dff").time.values[-1]
    ]

    if decode_dim == "time":
        assert (
            time_fractions is not None
        ), "time_fractions must be provided when decode_dim is 'time'"
        np.floor(len(stimulus_table) * time_fractions)
        return np.array(list(range(0, int(1 / time_fractions)))).repeat(
            int(np.floor(len(stimulus_table) * time_fractions))
        )  ## essentially bin trials into bins of time

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


def run_decoding(
    session_id,
    planes,
    stimulus_type_training,
    stimulus_type_testing,
    repetitions,
    decode_dim,
    chunk_range=None,
    folds=5,
    bootstrap=True,
    bootstrap_size=250,
    metrics_df=None,
    data_folder="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
    log=True,
    match_trials=False,
    time_fractions=None,
    subset_ns2=False,
):

    ############################## Prior to decoding, perform initialization tests ##############################
    # Set up logging if log_queue is provided (global variable)
    if log_queue is not None:
        setup_worker_logger()

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
    except:
        return None

    # Figure out chunk range if not given
    if chunk_range is None:
        chunk_range = (0, repetitions)

    # Check number of planes provided
    planes_in_session = session.get_planes()
    num_planes_in_session = len(planes_in_session)
    num_planes_provided = len(planes)
    if num_planes_in_session == 1 and num_planes_provided == 1:
        if planes[0] not in planes_in_session:
            logging.warning(
                f"Provided plane {planes[0]} is not valid for session {session_id}."
            )
            return None
        else:
            (
                logging.info(
                    f"{session_id}, {planes}: One plane provided, decoding will be performed on this plane."
                )
                if log
                else None
            )
    elif num_planes_in_session == 1 and num_planes_provided > 1:
        (
            logging.warning(
                f"{session_id}, {planes}: Only one plane in session, but multiple planes provided"
            )
            if log
            else None
        )
        return None
    else:
        (
            logging.info(f"{session_id}, {planes}: Planes provided are in session.")
            if log
            else None
        )

    # Check if ROIs are unduplicated or duplicated (i.e. 2p or 3p data, respectively)
    column = session.get_column_id()
    unduplicated = (
        True if column != 1 else False
    )  # Non-column 1 sessions are all 2p data (and ROIs are unduplicated)
    unduplicated = (
        False if column == 1 and len(session.get_planes()) == 1 else True
    )  # Column 1 3p data are unduplicated
    if column == 1 and len(session.get_planes()) > 1:
        (
            logging.warning(
                f"{session_id}: This is a 2p session in column 1, all ROIs are duplicated. Skipping decoding."
            )
            if log
            else None
        )
        return None

    # Set up decoding folder name based on parameters
    if tag:
        if match_trials:
            decoding_folder_name = os.path.join(
                results_folder,
                f"{tag}_TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_Boot{bootstrap_size}_Rep{repetitions}_NumPlanes{num_planes_provided}_MatchTrials",
            )
        else:
            decoding_folder_name = os.path.join(
                results_folder,
                f"{tag}_TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_Boot{bootstrap_size}_Rep{repetitions}_NumPlanes{num_planes_provided}",
            )
    else:
        if match_trials:
            decoding_folder_name = os.path.join(
                results_folder,
                f"TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}_NumPlanes{num_planes_provided}_MatchTrials",
            )
        else:
            decoding_folder_name = os.path.join(
                results_folder,
                f"TRAIN{stimulus_type_training}_TEST{stimulus_type_testing}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}_NumPlanes{num_planes_provided}",
            )
    if not os.path.exists(decoding_folder_name):
        os.makedirs(decoding_folder_name)

    # Create file name based on session ID and planes
    file_name = f"{session_id}_planes{planes}_reps{chunk_range[0]}-{chunk_range[1]}.pkl"
    file_name = os.path.join(decoding_folder_name, file_name)

    # Check if the decoding results df is already calculated and saved locally
    # if os.path.isfile(file_name):
    #     indiv_results_df = pd.read_pickle(file_name)
    #     (
    #         logging.info(f"{session_id}: Decoding results already exist. Returning df.")
    #         if log
    #         else None
    #     )
    #     return indiv_results_df

    ############################## Decoding starts here ##############################
    (
        logging.info(f"{session_id}: Starting decoding for planes {planes}")
        if log
        else None
    )

    # Load X data -- check if stimulus types are the same for training and testing
    X_data_training = pd.DataFrame()
    for p in planes:
        if len(X_data_training) == 0:
            # If X_data_training is empty, initialize it with the first plane's data
            X_data_training = get_X_data(
                session=session,
                plane=p,
                stimulus_type=stimulus_type_training,
                metrics_df=metrics_df,
                unduplicated=unduplicated,
                folder_name=data_folder,
            )
        else:
            # Concatenate the new plane's data to the existing X_data_training
            X_data_new = get_X_data(
                session=session,
                plane=p,
                stimulus_type=stimulus_type_training,
                metrics_df=metrics_df,
                unduplicated=unduplicated,
                folder_name=data_folder,
            )
            if len(X_data_new) > 0:
                # Only concatenate if the new data is not empty
                X_data_training = np.concatenate((X_data_training, X_data_new), axis=1)

    if stimulus_type_training == stimulus_type_testing:
        X_data_testing = None  # If training and testing stimulus types are the same, use the same X_data
    else:
        X_data_testing = pd.DataFrame()
        for p in planes:
            if len(X_data_testing) == 0:
                # If X_data_testing is empty, initialize it with the first plane's data
                X_data_testing = get_X_data(
                    session=session,
                    plane=p,
                    stimulus_type=stimulus_type_testing,
                    metrics_df=metrics_df,
                    unduplicated=unduplicated,
                    folder_name=data_folder,
                )
            else:
                # Concatenate the new plane's data to the existing X_data_testing
                X_data_new = get_X_data(
                    session=session,
                    plane=p,
                    stimulus_type=stimulus_type_testing,
                    metrics_df=metrics_df,
                    unduplicated=unduplicated,
                    folder_name=data_folder,
                )
                if len(X_data_new) > 0:
                    # Only concatenate if the new data is not empty
                    X_data_testing = np.concatenate(
                        (X_data_testing, X_data_new), axis=1
                    )

    if X_data_training is None:
        return None
    if np.shape(X_data_training)[1] < bootstrap_size:
        (
            logging.warning(
                f"{session_id}: Not enough X_data for training, only have {np.shape(X_data_training)[1]} rois, cannot perform decoding"
            )
            if log
            else None
        )
        return None
    if X_data_testing is None and stimulus_type_training != stimulus_type_testing:
        return None

    # Load Y data
    Y_data_training = get_Y_data(
        session=session,
        plane=planes[0],
        stimulus_type=stimulus_type_training,
        decode_dim=decode_dim,
        time_fractions=time_fractions,
    )
    if Y_data_training is None:
        return None

    if decode_dim == "time":
        while Y_data_training.shape[0] != X_data_training.shape[0]:  # should be true
            Y_data_training = np.append(
                Y_data_training, int(1 / time_fractions)
            )  # add a maximum of 4 values to the end to make it the same length as X_data

    assert (
        Y_data_training.shape[0] == X_data_training.shape[0]
    ), "Y_data and X_data (training) must be the same length"

    if stimulus_type_testing == stimulus_type_training:
        Y_data_testing = None  # If training and testing stimulus types are the same, use the same Y_data
    else:
        Y_data_testing = get_Y_data(
            session=session,
            plane=planes[0],
            stimulus_type=stimulus_type_testing,
            decode_dim=decode_dim,
            time_fractions=time_fractions,
        )
        if Y_data_testing is None:
            return None
        if decode_dim == "time":
            while Y_data_testing.shape[0] != X_data_testing.shape[0]:  # should be true
                Y_data_testing = np.append(
                    Y_data_testing, int(1 / time_fractions)
                )  # add a maximum of 4 values to the end to make it the same length as X_data
            assert (
                Y_data_testing.shape[0] == X_data_testing.shape[0]
            ), "Y_data and X_data (testing) must be the same length"

    # Subset the data so that both the training and testing data have the same directions / image indices
    if stimulus_type_training != stimulus_type_testing and match_trials:
        # (
        #     logging.info(
        #         f"{session_id}: Matching trials between training and testing data"
        #     )
        #     if log
        #     else None
        # )
        # Figure out which indices are in both y_data and y_data_other
        unique_trials_training = np.unique(Y_data_training)
        unique_trials_testing = np.unique(Y_data_testing)
        common_indices = np.intersect1d(unique_trials_training, unique_trials_testing)

        # (
        #     logging.info(
        #         f"{session_id}, {planes}: Original training trials: {np.shape(X_data_training)}, {len(Y_data_training)} and testing trials: {np.shape(X_data_testing)}, {len(Y_data_testing)}"
        #     )
        #     if log
        #     else None
        # )

        # Subset X_data and Y_data to only include common trial identities
        X_data_training = X_data_training[np.isin(Y_data_training, common_indices), :]
        X_data_testing = X_data_testing[np.isin(Y_data_testing, common_indices), :]
        Y_data_training = Y_data_training[np.isin(Y_data_training, common_indices)]
        Y_data_testing = Y_data_testing[np.isin(Y_data_testing, common_indices)]

        # (
        #     logging.info(
        #         f"{session_id}, {planes}: New training trials: {np.shape(X_data_training)}, {len(Y_data_training)} and testing trials: {np.shape(X_data_testing)}, {len(Y_data_testing)}"
        #     )
        #     if log
        #     else None
        # )

    # Subset NS Set 2 to the first 20% of trials
    if subset_ns2 == True:
        if stimulus_type_training == "natural_images_12":
            X_data_training = X_data_training[
                : int(np.shape(X_data_training)[0] * 0.2), :
            ]
            Y_data_training = Y_data_training[: int(len(Y_data_training) * 0.2)]
        if stimulus_type_testing == "natural_images_12":
            X_data_testing = X_data_testing[: int(np.shape(X_data_testing)[0] * 0.2), :]
            Y_data_testing = Y_data_testing[: int(len(Y_data_testing) * 0.2)]

    all_val_accuracies, all_test_accuracies, all_best_ks = [], [], []
    all_shuf_val_accuracies, all_shuf_test_accuracies, all_shuf_best_ks = [], [], []

    # Decoder loop -- perform decoding for each repetition # in chunk range
    for i in range(chunk_range[0], chunk_range[1]):
        if bootstrap:
            # Select X_data with replacement to control for different #s of neurons between recordings
            cols = np.random.choice(
                np.arange(X_data_training.shape[1]), size=bootstrap_size, replace=True
            )
            X_data_training_copy = X_data_training[:, cols]
            if X_data_testing is not None:
                X_data_testing_copy = X_data_testing[:, cols]
            else:
                X_data_testing_copy = None
        else:
            X_data_training_copy = X_data_training.copy()
            if X_data_testing is not None:
                X_data_testing_copy = X_data_testing.copy()
            else:
                X_data_testing_copy = None

        Y_data_training_copy = Y_data_training.copy()
        Y_data_testing_copy = (
            Y_data_testing.copy() if Y_data_testing is not None else None
        )

        for shuf_type in [False, True]:
            val_accuracies, test_accuracies, best_ks = (
                np.zeros(folds),
                np.zeros(folds),
                np.zeros(folds),
            )

            # Perform nested cross-validation
            for fold in range(folds):
                x_train, x_val, y_train, y_val = train_test_split(
                    X_data_training_copy,
                    Y_data_training_copy,
                    test_size=0.2,
                    random_state=int(
                        i * folds + fold + 1
                    ),  # set random state for reproducibility
                )

                if shuf_type:
                    # Shuffle the training data labels
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

                # Predict on held-out set
                y_pred_val = knn.predict(x_val)
                y_pred_test = (
                    knn.predict(X_data_testing_copy)
                    if X_data_testing_copy is not None
                    else None
                )

                # Calculate accuracy scores
                val_accuracy = metrics.accuracy_score(y_val, y_pred_val)
                test_accuracy = (
                    metrics.accuracy_score(Y_data_testing_copy, y_pred_test)
                    if y_pred_test is not None
                    else None
                )

                # Store accuracies and best k
                val_accuracies[fold] = val_accuracy
                test_accuracies[fold] = test_accuracy
                best_ks[fold] = best_k

                # Clear up some memory by deleting variables that are no longer needed
                del (
                    knn,
                    grid_search,
                    x_train,
                    x_val,
                    y_train,
                    y_val,
                    y_pred_val,
                    y_pred_test,
                    best_k,
                    val_accuracy,
                    test_accuracy,
                )
                gc.collect()

            # Save cross-validation scores, mean of scores, and std of scores
            if shuf_type:
                all_shuf_val_accuracies.append(val_accuracies)
                all_shuf_test_accuracies.append(test_accuracies)
                all_shuf_best_ks.append(best_ks)
            else:
                all_val_accuracies.append(val_accuracies)
                all_test_accuracies.append(test_accuracies)
                all_best_ks.append(best_ks)

            # Clear up some memory by deleting variables that are no longer needed
            del (val_accuracies, test_accuracies, best_ks)

    # Save info about recording + decoding #
    repetition_nums = np.tile(np.arange(repetitions), chunk_range[1] - chunk_range[0])

    # Create a DataFrame to store the results
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                repetition_nums,
                all_val_accuracies,
                all_test_accuracies,
                all_best_ks,
                all_shuf_val_accuracies,
                all_shuf_test_accuracies,
                all_shuf_best_ks,
            )
        ),
        columns=[
            "repetition_num",
            "val_accuracy",
            "test_accuracy",
            "best_k",
            "shuf_val_accuracy",
            "shuf_test_accuracy",
            "shuf_best_k",
        ],
    )
    indiv_results_df["mouse_id"] = session.get_mouse_id()
    indiv_results_df["column_id"] = session.get_column_id()
    indiv_results_df["volume_id"] = session.get_volume_id()
    indiv_results_df["plane_id"] = "".join([str(p) for p in planes])

    # Save the results DataFrame to a pickle file
    if save_decoding:
        indiv_results_df.to_pickle(file_name)
        (
            logging.info(
                f"{session_id}, {planes}, {chunk_range}: Decoding results saved to {file_name}"
            )
            if log
            else None
        )

    return indiv_results_df


if __name__ == "__main__":
    # Load parameters from the command line or set them here
    parser = argparse.ArgumentParser(
        prog="V1DD_Decoding", description="Decoding parameters"
    )
    parser.add_argument("--tag", type=str, default=None, help="Tag for saving results")
    parser.add_argument(
        "--repetitions", type=int, default=1000, help="Number of bootstrap repetitions"
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        help="If provided, perform bootstrap sampling with replacement",
    )
    parser.add_argument(
        "--bootstrap_size",
        type=int,
        default=250,
        help="Size of bootstrap sample or threshold for minimum number of ROIs",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="Size of chunks for parallel processing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--stim_train",
        type=str,
        help="Stimulus type for training; options are: drifting_gratings_full, drifting_gratings_windowed, natural_images_12, natural_images",
    )
    parser.add_argument(
        "--stim_test",
        type=str,
        default=None,
        help="Stimulus type for testing; options are: drifting_gratings_full, drifting_gratings_windowed, natural_images_12, natural_images",
    )
    parser.add_argument(
        "--one_plane",
        action=argparse.BooleanOptionalAction,
        help="If provided, decode only one plane; else, decode multiple planes (usually 3)",
    )
    parser.add_argument(
        "--match_trials",
        action=argparse.BooleanOptionalAction,
        help="If provided, match trials between training and testing datasets",
    )
    parser.add_argument(
        "--decode_time",
        action=argparse.BooleanOptionalAction,
        help="If provided, decode time within a stimulus",
    )
    parser.add_argument(
        "--subset_ns2",
        action=argparse.BooleanOptionalAction,
        help="Just use if train/testing on natural images and want to subset the NS2 test data to the first 20%",
    )
    args = parser.parse_args()

    # Set up logging
    log_dir = os.path.join(ARTIFACT_DIR, "decoding_analyses/logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = (
        os.path.join(log_dir, f"decoding_{args.tag}_{timestamp}.log")
        if args.tag is not None
        else os.path.join(log_dir, f"decoding_{timestamp}.log")
    )
    manager = Manager()
    log_queue = manager.Queue()
    file_handler = listener_config(log_file)
    listener = QueueListener(log_queue, file_handler)
    listener.start()

    # Set up main logger to also use the queue
    setup_main_logger(log_queue)

    decode_dims = {
        "drifting_gratings_full": "direction",
        "drifting_gratings_windowed": "direction",
        "natural_images_12": "image_index",
        "natural_images": "image_index",
    }
    if args.decode_time is not None:
        decode_dims["drifting_gratings_full"] = "time"
        decode_dims["drifting_gratings_windowed"] = "time"
        decode_dims["natural_images_12"] = "time"
        decode_dims["natural_images"] = "time"

    multi_stim_pairs = {
        "drifting_gratings_full": "drifting_gratings_windowed",
        "drifting_gratings_windowed": "drifting_gratings_full",
        "natural_images_12": "natural_images",
        "natural_images": "natural_images_12",
    }

    ## Decoding parameters
    tag = args.tag
    repetitions = args.repetitions
    bootstrap_size = args.bootstrap_size
    bootstrap = True if args.bootstrap is not None else False
    one_plane = True if args.one_plane is not None else False
    data_folder = "/home/naomi/Desktop/data"
    path_to_nwbs = f"{data_folder}/V1dd_nwbs"
    path_to_metrics = f"{data_folder}/all_metrics_240426.csv"
    results_folder = f"{data_folder}/decoding_results"
    log = args.verbose
    match_trials = True if args.match_trials is not None else False
    subset_ns2 = True if args.subset_ns2 is not None else False

    chunk_size = args.chunk_size
    if repetitions == 1:
        chunk_size = 1

    if repetitions % chunk_size != 0:
        raise ValueError(
            "Repetitions must be divisible by chunk size for parallel processing."
        )
    repetition_chunks = [(i, i + chunk_size) for i in range(0, repetitions, chunk_size)]

    if one_plane == True:
        plane_list = [[1], [2], [3], [4], [5], [6]]
    else:
        plane_list = [[1, 2, 3], [4, 5, 6]]

    stim_train = args.stim_train
    stim_test = args.stim_test
    if stim_test is None:
        stim_test = stim_train

    if stim_train not in decode_dims.keys():
        raise ValueError(f"Training stimulus type {stim_train} is not valid.")
    if stim_test not in decode_dims.keys():
        raise ValueError(f"Testing stimulus type {stim_test} is not valid.")

    if stim_train != stim_test:
        if stim_test != multi_stim_pairs[stim_train]:
            raise ValueError(
                f"Training with {stim_train} and testing with {stim_test} is not valid."
                f"Options for testing with {stim_train} are: {multi_stim_pairs[stim_train]}"
            )

    ## Load in the client
    client = OPhysClient(path_to_nwbs)

    ## Load in metrics
    metrics_df = pd.read_csv(path_to_metrics)

    print(
        "Starting decoding with the following parameters:"
        f"\nTag: {tag}"
        f"\nRepetitions: {repetitions}"
        f"\nBootstrap: {bootstrap}"
        f"\nBootstrap Size: {bootstrap_size}"
        f"\nChunk Size: {chunk_size}"
        f"\nStimulus Type Training: {stim_train}"
        f"\nStimulus Type Testing: {stim_test}"
        f"\nOne Plane: {one_plane}"
        f"\nPlane List: {plane_list}"
        f"\nMatch Trials: {match_trials}"
    )
    logging.info(
        f"Decoding parameters: {repetitions} repetitions, {bootstrap_size} bootstrap size, tag: {tag}, bootstrap, {bootstrap}, one_plane: {one_plane}"
    )

    ## Apply multiprocessing to run decoding in parallel
    Parallel(n_jobs=30)(
        delayed(run_decoding)(
            session_id=session_id,
            planes=planes,
            stimulus_type_training=stim_train,
            stimulus_type_testing=stim_test,
            repetitions=repetitions,
            decode_dim=decode_dims[stim_train],
            chunk_range=chunk_range,
            folds=5,
            bootstrap=bootstrap,
            bootstrap_size=bootstrap_size,
            metrics_df=metrics_df,
            data_folder=data_folder,
            save_decoding=True,
            results_folder=results_folder,
            tag=tag,
            log=True,
            match_trials=match_trials,
            time_fractions=0.25 if args.decode_time is not None else None,
            subset_ns2=subset_ns2,
        )
        for session_id in client.get_all_session_ids()
        for planes in plane_list
        for chunk_range in repetition_chunks
    )

    listener.stop()

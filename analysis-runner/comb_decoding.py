import gc
import os
import argparse
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
):

    ############################## Prior to decoding, perform initialization tests ##############################
    # Set up logging if log_queue is provided (global variable)
    if log_queue is not None:
        setup_worker_logger()

    # Load the session using the client
    try:
        session = client.load_ophys_session(session_id)
        logging.info(f"{session_id}: Loaded") if log else None
    except Exception as e:
        logging.error(f"Error loading session {session_id}: {e}") if log else None
        return None

    # Figure out chunk range if not given
    if chunk_range is None:
        chunk_range = (0, repetitions)

    # Check number of planes provided
    planes = [p for p in planes if p in session.get_planes()]
    num_planes = len(planes)
    if num_planes == 1:
        plane = planes[0]
    elif num_planes > 1:
        plane = None
    elif num_planes == 0:
        (
            logging.error(f"Provided planes are not valid for session {session_id}.")
            if log
            else None
        )
        return None

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
            logging.info(
                f"{session_id}: This is a 2p session in column 1, all ROIs are duplicated. Skipping decoding."
            )
            if log
            else None
        )
        return None

    # Set up decoding folder name based on parameters
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
    if not os.path.exists(decoding_folder_name):
        os.makedirs(decoding_folder_name)

    # Create file name based on session ID and planes
    file_name = f"{session_id}_planes{planes}.pkl"
    file_name = os.path.join(decoding_folder_name, file_name)

    # Check if the decoding results df is already calculated and saved locally
    if os.path.isfile(file_name):
        indiv_results_df = pd.read_pickle(file_name)
        (
            logging.info(f"{session_id}: Decoding results already exist. Returning df.")
            if log
            else None
        )
        return indiv_results_df

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
            X_data_training = np.concatenate(
                (
                    X_data_training,
                    get_X_data(
                        session=session,
                        plane=p,
                        stimulus_type=stimulus_type_training,
                        metrics_df=metrics_df,
                        unduplicated=unduplicated,
                        folder_name=data_folder,
                    ),
                ),
                axis=1,
            )
    if stimulus_type_training == stimulus_type_testing:
        X_data_testing = None  # If training and testing stimulus types are the same, use the same X_data
    else:
        X_data_testing = pd.DataFrame()
        for p in planes:
            if len(X_data_testing) == 0:
                # If X_data_testing is empty, initialize it with the first plane's data
                X_data_testing = get_X_data(
                    session=session,
                    plane=plane,
                    stimulus_type=stimulus_type_training,
                    metrics_df=metrics_df,
                    unduplicated=unduplicated,
                    folder_name=data_folder,
                )
            else:
                # Concatenate the new plane's data to the existing X_data_testing
                X_data_testing = np.concatenate(
                    (
                        X_data_testing,
                        get_X_data(
                            session=session,
                            plane=plane,
                            stimulus_type=stimulus_type_training,
                            metrics_df=metrics_df,
                            unduplicated=unduplicated,
                            folder_name=data_folder,
                        ),
                    ),
                    axis=1,
                )

    if X_data_training is None:
        return None
    if np.shape(X_data_training)[1] < bootstrap_size:
        (
            logging.info(
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
    )
    if type(Y_data_training[0]) is not int:
        encoder = preprocessing.LabelEncoder()
        Y_data_training = encoder.fit_transform(Y_data_training)
    if Y_data_training is None:
        return None

    if stimulus_type_testing == stimulus_type_training:
        Y_data_testing = None  # If training and testing stimulus types are the same, use the same Y_data
    else:
        Y_data_testing = get_Y_data(
            session=session,
            plane=planes[0],
            stimulus_type=stimulus_type_testing,
            decode_dim=decode_dim,
        )
        if type(Y_data_testing[0]) is not int:
            encoder = preprocessing.LabelEncoder()
            Y_data_testing = encoder.fit_transform(Y_data_testing)
        if Y_data_testing is None:
            return None

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
    mouse_ids = np.tile(session.get_mouse_id(), chunk_range[1] - chunk_range[0])
    column_ids = np.tile(session.get_column_id(), chunk_range[1] - chunk_range[0])
    volume_ids = np.tile(session.get_volume_id(), chunk_range[1] - chunk_range[0])
    planes = np.tile(planes, chunk_range[1] - chunk_range[0])
    repetition_nums = np.tile(np.arange(repetitions), chunk_range[1] - chunk_range[0])

    # Create a DataFrame to store the results
    indiv_results_df = pd.DataFrame(
        data=list(
            zip(
                mouse_ids,
                column_ids,
                volume_ids,
                planes,
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
            "mouse_id",
            "column_id",
            "volume_id",
            "planes",
            "repetition_num",
            "val_accuracy",
            "test_accuracy",
            "best_k",
            "shuf_val_accuracy",
            "shuf_test_accuracy",
            "shuf_best_k",
        ],
    )

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
    parser = argparse.ArgumentParser(description="Decoding parameters")
    parser.add_argument("--tag", type=str, default=None, help="Tag for saving results")
    parser.add_argument(
        "--repetitions", type=int, default=1000, help="Number of bootstrap repetitions"
    )
    parser.add_argument(
        "--bootstrap",
        type=bool,
        default=True,
        help="If True, perform bootstrap sampling; if False, use all ROIs in the session",
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
        default=25,
        help="Size of chunks for parallel processing",
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=True, help="Enable verbose logging"
    )
    parser.add_argument(
        "--stim_train",
        type=str,
        help="Stimulus type for training; options are: drifting_gratings_full, drifting_gratings_windowed, natural_images_12, natural_images",
    )
    parser.add_argument(
        "--stim_test",
        type=str,
        help="Stimulus type for testing; options are: drifting_gratings_full, drifting_gratings_windowed, natural_images_12, natural_images",
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

    ## Decoding parameters
    tag = args.tag
    repetitions = args.repetitions
    bootstrap_size = args.bootstrap_size
    bootstrap = args.bootstrap
    data_folder = "/home/naomi/Desktop/data"
    path_to_nwbs = f"{data_folder}/V1dd_nwbs"
    path_to_metrics = f"{data_folder}/all_metrics_240426.csv"
    results_folder = f"{data_folder}/decoding_results"

    chunk_size = args.chunk_size
    if repetitions % chunk_size != 0:
        raise ValueError(
            "Repetitions must be divisible by chunk size for parallel processing."
        )
    repetition_chunks = [(i, i + chunk_size) for i in range(0, repetitions, chunk_size)]
    stim_train = args.stim_train
    stim_test = args.stim_test
    if stim_train not in stim_types:
        raise ValueError(f"Training stimulus type {stim_train} is not valid.")
    if stim_test not in stim_types:
        raise ValueError(f"Testing stimulus type {stim_test} is not valid.")

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
    )
    logging.info(
        f"Decoding parameters: {repetitions} repetitions, {bootstrap_size} bootstrap size, tag: {tag}, bootstrap, {bootstrap}, one_plane: {one_plane}, multi_stim: {multi_stim}"
    )

    # apply multiprocessing to run decoding in parallel

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

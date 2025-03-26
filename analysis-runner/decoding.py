import os, random, sys, math
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from itertools import compress
from allen_v1dd.client import OPhysClient

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold


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
                & (metrics_df["plane"] == plane)
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
    planes,
    stimulus_type,
    metrics_df=None,
    unduplicated=False,
    folder_name="/home/naomi/Desktop/data",
):

    X_data_df = pd.DataFrame()
    for plane in planes:
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


def get_Y_data(session, stimulus_type, decode_dim):

    stimulus_table, _ = session.get_stimulus_table(stimulus_type)
    stimulus_table = stimulus_table.dropna()

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
    session,
    planes,
    stimulus_type,
    repetitions,
    decode_dim,
    cross_validation=True,
    folds=5,
    max_neighbors=15,
    metric="correlation",
    test_size=0.2,
    bootstrap=True,
    bootstrap_size=250,
    metrics_df=None,
    unduplicated=False,
    folder_name="/home/naomi/Desktop/data",
    save_decoding=True,
    results_folder="/home/naomi/Desktop/data/decoding_results",
    tag=None,
):
    if tag:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{tag}_{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}_UndupROIs{unduplicated}_CV{cross_validation}",
        )
    else:
        decoding_folder_name = os.path.join(
            results_folder,
            f"{stimulus_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}_UndupROIs{unduplicated}_CV{cross_validation}",
        )

    file_name = f"{get_session_id(session)}_{planes[0]}{planes[1]}{planes[2]}.pkl"

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
        session, planes, stimulus_type, metrics_df, unduplicated, folder_name
    )
    if X_data is None:
        print("Cannot find any X data...cannot perform decoding")
        return None
    if np.shape(X_data)[1] < 2:
        print("Not enough X_data...cannot perform decoding")
        return None

    # Check to see if there is Y_data (i.e. if there are corresponding visual stimuli for this session)
    Y_data = get_Y_data(session, stimulus_type, decode_dim)
    if Y_data is None:
        return None

    for i in range(repetitions):

        # Load in X_data (neural responses)
        X_data = get_X_data(
            session, planes, stimulus_type, metrics_df, unduplicated, folder_name
        )

        if (
            bootstrap
        ):  # select w/ replacement -- to control for different #s of neurons between recs.
            X_data = (
                pd.DataFrame(X_data).T.sample(n=bootstrap_size, replace=True).T.values
            )

        # Load in Y_data (visual stimulus identity)
        Y_data = get_Y_data(session, stimulus_type, decode_dim)

        # Check type of Y_data and make sure it is able to be decoded
        # (i.e. has to all be discrete integer values, not continuous)
        if type(Y_data[0]) is not int:
            lab_enc = preprocessing.LabelEncoder()
            Y_data = lab_enc.fit_transform(Y_data)

        # Figure out best # of neighbors to use
        knn, best_k = knn_cross_validation(max_neighbors, metric, X_data, Y_data)

        # Calculate 5-fold cross validation scores
        fold_scores = cross_val_score(knn, X_data, Y_data, cv=5)

        # Save cross-val scores, mean of scores, and std of scores
        test_accuracies.append(fold_scores)
        test_accuracies_mean.append(np.mean(fold_scores))
        test_accuracies_std.append(np.std(fold_scores))
        num_k_neighbors.append(best_k)

        # Shuffle data and perform same decoding for comparison
        X_data = np.random.permutation(X_data)

        # Figure out best # of neighbors to use
        knn, best_k = knn_cross_validation(max_neighbors, metric, X_data, Y_data)

        # Calculate 5-fold cross validation scores
        fold_scores = cross_val_score(knn, X_data, Y_data, cv=5)

        # Save cross-val scores, mean of scores, and std of scores
        shuf_test_accuracies.append(fold_scores)
        shuf_test_accuracies_mean.append(np.mean(fold_scores))
        shuf_test_accuracies_std.append(np.std(fold_scores))
        shuf_num_k_neighbors.append(best_k)

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
            os.path.join(
                decoding_folder_name,
                f"{get_session_id(session)}_{planes[0]}{planes[1]}{planes[2]}.pkl",
            ),
        )

    return indiv_results_df


def decoding_wrapper(input):
    # Take input and break up into individual inputs to run the decoding
    run_decoding(
        session=input[0],
        planes=input[1],
        stimulus_type=input[2],
        repetitions=input[3],
        decode_dim=input[4],
        cross_validation=input[5],
        folds=input[6],
        max_neighbors=input[7],
        metric=input[8],
        test_size=input[9],
        bootstrap=input[10],
        bootstrap_size=input[11],
        metrics_df=input[12],
        unduplicated=input[13],
        folder_name=input[14],
        save_decoding=input[15],
        results_folder=input[16],
        tag=input[17],
    )

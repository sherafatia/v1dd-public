import os
import pandas as pd
from decoding import *
from v1dd_public import ARTIFACT_DIR
from joblib import Parallel, delayed
from allen_v1dd.client import OPhysClient


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

### Create logging file


import logging
from datetime import datetime
log_dir = os.path.join(ARTIFACT_DIR, "decoding_analyses/logs")
log_queue, listener = start_log(log_dir)
setup_worker_logger(log_queue)

## Decoding parameters
repetitions = 1000  # Number of bootstrap repetitions
bootstrap = True  # If True, perform bootstrap sampling; if False, use all ROIs in the session
bootstrap_size = 250  # Size of bootstrap sample; or threshold for minimum number of ROIs to perform decoding (if bootstrap=False)
tag = "2025_0728_4"  # Tag for saving results -- will take form "{tag}_{stim_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}"
data_folder = "/home/naomi/Desktop/data"
path_to_nwbs = f"{data_folder}/V1dd_nwbs"
path_to_metrics = f"{data_folder}/all_metrics_240426.csv"
results_folder = f"{data_folder}/decoding_results"
one_plane = False  # If True, decode across a single plane; if False, decode across multiple planes (e.g. just 2p data)

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
    "drifting_gratings_full": ["direction"],
    "drifting_gratings_windowed": ["direction"],
    "natural_images_12": ["image_index"],
    "natural_images": ["image_index"],
}

logging.info(f"Decoding parameters: {repetitions} repetitions, {bootstrap_size} bootstrap size, tag: {tag}, bootstrap, {bootstrap}, one_plane: {one_plane}")

# apply multiprocessing to run decoding in parallel
if one_plane:
    Parallel(n_jobs=-1)(
        delayed(run_decoding_one_plane)(
            session_id=session_id,
            plane=plane,
            stimulus_type=stim_type,
            repetitions=repetitions,
            decode_dim=dim,
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
        for dim in decode_dims[stim_type]
    )
else:
    Parallel(n_jobs=-1)(
        delayed(run_decoding_across_planes)(
            session_id=sess,
            planes=planes,
            stimulus_type=stim_type,
            repetitions=repetitions,
            decode_dim=dim,
            bootstrap=bootstrap,
            bootstrap_size=bootstrap_size,
            metrics_df=metrics_df,
            folder_name=data_folder,
            save_decoding=True,
            results_folder=results_folder,
            tag=tag,
            log=True,
        )
        for sess in client.get_all_session_ids()
        for planes in [[1, 2, 3], [4, 5, 6]]
        for stim_type in stim_types
        for dim in decode_dims[stim_type]
    )

# Convert results to DataFrame
all_results_df = pd.DataFrame()
for stim_type in stim_types:
    decode_dim = decode_dims[stim_type][
        0
    ]  # Assuming only one decode dimension per stim type

    path_name = f"/home/naomi/Desktop/data/decoding_results/{tag}_{stim_type}_{decode_dim}_Boot1_Rep1"
    results_df = pd.DataFrame()
    for filename in os.listdir(path_name):
        f = os.path.join(path_name, filename)
        results_df = pd.concat([results_df, pd.read_pickle(f)])

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
    ARTIFACT_DIR, f"decoding_analyses/{tag}_Boot{bootstrap_size}_Reps{repetitions}.pkl"
)
all_results_df.to_pickle(all_results_path)
logging.info(f"Saved all results in a dataframe: {all_results_path}")

stop_log(log_queue, listener)
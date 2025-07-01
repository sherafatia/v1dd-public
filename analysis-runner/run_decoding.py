import numpy as np
import pandas as pd
from decoding import get_mean_dff_traces, run_decoding
from allen_v1dd.client import OPhysClient
from joblib import Parallel, delayed

## Decoding parameters
repetitions = 1  # Number of bootstrap repetitions
bootstrap_size = 5  # Size of bootstrap sample; or threshold for minimum number of ROIs to perform decoding (if bootstrap=False)
tag = "2025_0701_2"  # Tag for saving results -- will take form "{tag}_{stim_type}_{decode_dim}_Boot{bootstrap_size}_Rep{repetitions}"
data_folder = "/home/naomi/Desktop/data"
path_to_nwbs = f"{data_folder}/V1dd_nwbs"
path_to_metrics = f"{data_folder}/all_metrics_240426.csv"
results_folder = f"{data_folder}/decoding_results"

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
    "drifting_gratings_full": ["direction", "spatial"],
    "drifting_gratings_windowed": ["direction", "spatial"],
    "natural_images_12": ["image_index"],
    "natural_images": ["image_index"],
}

# Iterate through all sessions and perform decoding
for sess_id in client.get_all_session_ids():
    try:
        sess = client.load_ophys_session(sess_id)
    except ValueError as e:
        print(f"Error loading session {sess_id}: {e}")
        continue
    column = sess.get_column_id()
    num_planes = len(sess.get_planes())

    # Figure out if we need to choose "unduplicated" ROIs -- note all ROIs in column 1 are set to "duplicated", but only the 2p data is actually duplicated
    # unduplicated = True --> use unduplicated ROIs for decoding (e.g. 2p data, not from column 1)
    # unduplicated = False --> use duplicated ROIs for decoding (e.g. 3p data)
    if column != 1:
        undup = True
    elif column == 1 and num_planes > 1:
        continue  # skip all 2p sessions in column 1 -- these have all duplicated ROIs
    elif column == 1 and num_planes == 1:
        undup = False

    print(f"Performing decoding for session {sess_id}")

    # apply multiprocessing to run decoding in parallel
    Parallel(n_jobs=-1)(
        delayed(run_decoding)(
            session=sess,
            plane=plane,
            stimulus_type=stim_type,
            repetitions=repetitions,
            decode_dim=dim,
            max_neighbors=15,
            metric="correlation",
            bootstrap=False,
            bootstrap_size=bootstrap_size,
            metrics_df=metrics_df,
            unduplicated=undup,
            folder_name=data_folder,
            save_decoding=True,
            results_folder=results_folder,
            tag=tag,
        )
        for plane in sess.get_planes()
        for stim_type in stim_types
        for dim in decode_dims[stim_type]
    )

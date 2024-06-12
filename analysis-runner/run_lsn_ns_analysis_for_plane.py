from argparse import ArgumentParser
from datetime import date
import logging
import time
from typing import List, Optional
from joblib import Parallel, delayed
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from allen_v1dd.client import OPhysClient
from v1dd_public import ARTIFACT_DIR
import h5py
import warnings

from v1dd_public.utils import compute_lsn_ns_metrics_for_col_vol_plane
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

DATA_DIR = Path("/home/roozbehf/Documents/v1dd_arefeh/V1_DD_NWBs/")
NWB_DATA_DIR = DATA_DIR / "nwbs"

def run_analysis_for_plane(analysis_name: str,
                    start_idx: int = 0,
                    trace_type: str = "events"):
    
    logger.info("Running analysis...")
    start_time = time.perf_counter()
    compute_metrics_for_all_mice(analysis_name,
                                    start_idx=start_idx,
                                    trace_type = trace_type)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"Analysis took {total_time:.4f} seconds.")
    
def compute_metrics_for_all_mice(analysis_name: str,
                                    start_idx: int = 0,
                                    n_jobs: Optional[int] = -1,
                                    trace_type: str = "events"):
    client =  OPhysClient(DATA_DIR)
    nwb_file_paths = list(NWB_DATA_DIR.glob("processed/*.nwb"))
    
    data = create_input_data(client=client, 
                                nwb_file_paths=nwb_file_paths, 
                                start_idx=start_idx)
            
    all_metrics = Parallel(n_jobs=n_jobs, verbose=1)(delayed(compute_lsn_ns_metrics_for_col_vol_plane)(client, d["mouse_id"], d["col_vol_id"], d["plane"]) for d in data)
    logger.info(f"Done computing {len(all_metrics)} metrics.")
    
    h5_name = save_data_to_h5_file(all_metrics, analysis_name)
    logger.info(f"Saved H5 file to {h5_name.resolve()}")
    

def save_data_to_h5_file(all_metrics: List[dict], analysis_name: str) -> Path:
    """Save data to an H5 file."""
    h5_name = ARTIFACT_DIR / f"lsn_ns_metrics_{analysis_name}.h5"
    
    with h5py.File(h5_name, 'w') as h5:
        for metrics in tqdm(all_metrics, desc="Saving metrics"):
            mouse_id = metrics["mouse_id"]
            col_vol_id = metrics["col_vol"]
            plane_id = metrics["plane"] - 1
            
            metrics_data = metrics["data"]
            
            if metrics_data:             
                group = h5.require_group(mouse_id)
                subgroup1 = group.require_group(col_vol_id)      
                subgroup2 = subgroup1.require_group(f"Plane_{plane_id}")

                for h5_group_name in metrics_data.keys():
                    value = metrics_data[h5_group_name]
                    if isinstance(value, np.ndarray): 
                        subgroup2.create_dataset(h5_group_name, data = value)
                    else:
                        subgroup2.attrs[h5_group_name] = value
            else:
                logger.info(f"Skipping an empty metrics for {mouse_id} {col_vol_id} {plane_id}")
                
    return h5_name

def create_input_data(client: OPhysClient, 
                         nwb_file_paths: List[Path], 
                         start_idx: int = 0) -> list:
    """Create input data."""
    data = []
    for path in nwb_file_paths[start_idx:]:
        mouse_id, col_vol_id = path.stem.split("_")[:2]
        session_name = f"{mouse_id}_{col_vol_id}"
        session = client.load_ophys_session(session_name)
        n_planes = len(session.get_planes())
        for plane_id in range(n_planes):
            data.append({
                "mouse_id": mouse_id,
                "col_vol_id": col_vol_id,
                "plane": plane_id + 1
            })
    return data


def _create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run analysis for session.")
    parser.add_argument('-a', '--analysis-name', dest="analysis_name", type=str, default=str(date.today()), help="Tag for the name of H5 files")
    parser.add_argument('-s', '--start-idx', dest="start_idx", type=int, default=0, help="Session index to begin with")
    parser.add_argument('-t', '--trace-type', dest="trace_type", type=str, default="events", help="Trace type dff or events")
    return parser
                            
if __name__ == "__main__":
    parser = _create_parser()
    args = parser.parse_args() 
    run_analysis_for_plane(args.analysis_name, 
                                start_idx=args.start_idx,
                                trace_type=args.trace_type)
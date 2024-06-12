from argparse import ArgumentParser
import logging
from pathlib import Path
import pandas as pd
from v1dd_public import ARTIFACT_DIR
import h5py
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

MOUSE_ID_TO_MOUSE_CRE = {'M409828':'slc2', 'M438833':'slc4', 'M427836':'slc5', 'M416296':'teto1'}
            
key_names = ['valid_cell_index',
'cell_index',
'col_vol',
'column',
'volume',
'plane',
'x',
'y',
'z',
'2p3p',
'on_score',
'off_score',
'on_center_x_orig',
'on_center_y_orig',
'off_center_x_orig',
'off_center_y_orig',
'on_center_x',
'on_center_y',
'off_center_x',
'off_center_y',
'on_center_wx',
'on_center_wy',
'off_center_wx',
'off_center_wy',
"on_center_h",
"off_center_h",
'on_area',
'off_area',
'on_averaged_response_at_receptive_field',
'off_averaged_response_at_receptive_field',
'percentage_res_trial_4_locally_sparse_noise',
'frac_res_trial_4_locally_sparse_noise',
'frac_res_to_on',
'frac_res_to_off',
'total_responsive_trials_all_pixels',
"mu_spont",
"max_spont",
"min_spont",
"is_responsive",
"has_rf_mean_std",
"has_rf_chi2",
"max_n_responsive_trials_on",
"is_responsive_to_on",
"has_rf_mean_std_on",
"has_rf_v2_on",
"has_rf_zscore_on",
"max_n_responsive_trials_off",
"is_responsive_to_off",
"has_rf_mean_std_off",
"has_rf_v2_off",
"has_rf_zscore_off",
"has_on_rf",
"has_off_rf",
"on_angle",
"off_angle",
"on_angle_degree",
"off_angle_degree",
'frac_res_to_ns12',
'frac_res_to_ns118'
]

def make_csv_from_h5(mouse_ids: list[str],
                     key_names: list[str],
                     h5_dir: Path,
                     h5_tag: str,
                     csv_dir: Path) -> None:
    
    for mouse_id in mouse_ids:
        logger.info(f"Generating CSV for mouse ID '{mouse_id}'...")
        h5_name = h5_dir / f"lsn_ns_metrics_{h5_tag}.h5"

        key_name_to_vals = {k: [] for k in key_names}

        with h5py.File(h5_name, 'r+') as h5: 
            col_vols = h5[mouse_id].keys()
            
            for col_vol_id in col_vols:      
                col_vol_info = h5[mouse_id][col_vol_id]
                for plane_id in range(6):
                    if f"Plane_{plane_id}" in col_vol_info.keys():
                        plane_info = h5[mouse_id][col_vol_id][f"Plane_{plane_id}"]
                        for key_name in key_names:
                            n_val_cells = h5[mouse_id][col_vol_id][f"Plane_{plane_id}"]["2p3p"][:].shape[0]
                            if key_name == 'col_vol':
                                temp = np.full(n_val_cells, col_vol_id)
                                for value in temp:     
                                    key_name_to_vals[key_name].append(value)
                            elif key_name == 'column':
                                temp = np.full(n_val_cells, col_vol_id[0])
                                for value in temp:     
                                    key_name_to_vals[key_name].append(value)                                
                            elif key_name == 'volume':
                                temp = np.full(n_val_cells, col_vol_id[1])
                                for value in temp:     
                                    key_name_to_vals[key_name].append(value)
                            elif key_name == 'plane':
                                temp = np.full(n_val_cells, plane_id)
                                for value in temp:     
                                    key_name_to_vals[key_name].append(value)
                            else:
                                for value in plane_info[key_name]:     
                                    key_name_to_vals[key_name].append(value)

        mouse_rf_df = pd.DataFrame(key_name_to_vals)
        
        mouse_cre = MOUSE_ID_TO_MOUSE_CRE[mouse_id]
        csv_path = csv_dir / f'{mouse_cre}_{h5_tag}_lsn_ns_metrics.csv'
        mouse_rf_df.to_csv(csv_path)
        logger.info(f"Created CSV {csv_path}")

def _create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate Metric CSV Files.")
    parser.add_argument("-m", "--mouse_ids", nargs="+", 
                        default=list(MOUSE_ID_TO_MOUSE_CRE.keys()),
                        help="A list of mouse IDs.", type=str)
    parser.add_argument("--h5-dir", dest="h5_dir", default=ARTIFACT_DIR, help="Directory of the H5 files to load")
    parser.add_argument("--h5-tag", dest="h5_tag", default='', help="Tag of the H5 files to load")
    parser.add_argument("--csv-dir", dest="csv_dir", default=ARTIFACT_DIR, help="Directory of the output CSV files. Defaults to h5_dir.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                        help="A dry run of the CSV generation.")
    return parser

if __name__ == "__main__":
    parser = _create_parser()
    args = parser.parse_args() 
    
    h5_dir = Path(args.h5_dir)
    h5_tag = Path(args.h5_tag)
    csv_dir = Path(args.csv_dir)
    
    # Check valid directories.
    if not h5_dir.is_dir():
        raise NotADirectoryError(f"directory path '{h5_dir}' is not a directory.")

    csv_dir.mkdir(exist_ok=True)

    logger.info(f"Going to create CSVs for mouse IDs {args.mouse_ids} using H5 dir {h5_dir.resolve()} and CSV dir {csv_dir.resolve()}.")
    if not args.dry_run:
        make_csv_from_h5(args.mouse_ids, key_names, h5_dir=h5_dir, h5_tag=h5_tag, csv_dir=csv_dir)

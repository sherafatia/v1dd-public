import logging
import numpy as np
from scipy import stats
from allen_v1dd.client import OPhysClient
from allen_v1dd.client.ophys_session import OPhysSession

from allen_v1dd.stimulus_analysis import LocallySparseNoise
from allen_v1dd.stimulus_analysis.natural_scenes import NaturalScenes

from abbasilab_v1dd.locally_sparse_noise.chi_square_continuous import *
rng = np.random.default_rng(1)
from torch.functional import F
import torch
from torch import nn


def is_cell_responsive(n_responsive_trials, weighted_avg, min_responsive_trials = 7, nstd = 3):
    """
    This function determines if a cell is responsive to the LSN or not
    """
    subfield_responsive_trials = n_responsive_trials
    mean_responsive_trials = subfield_responsive_trials.mean()
    std_responsive_trials = subfield_responsive_trials.std()
    max_value_responsive_trials = subfield_responsive_trials.max()
   
    subfield_all_trials = weighted_avg
    mean_all_trials = subfield_all_trials.mean()
    std_all_trials = subfield_all_trials.std()
    max_value_all_trials = subfield_all_trials.max()
   
    if max_value_responsive_trials >= min_responsive_trials and \
    max_value_responsive_trials >=  mean_responsive_trials + nstd*std_responsive_trials and \
    max_value_all_trials >=  mean_all_trials + nstd*std_all_trials:
        return True
    else:
        return False
    
def make_sample_spontanous(n_cells,
                           lsn,
                           times,
                           dff,
                           tot_lsn_frames,
                           n_tot_pres,
                           rng,
                           sample_spontaneous: int = 1000) -> np.ndarray:
    
    # define a sample of 1000 df/f values for the spontaneous activity
    spont_vals = np.zeros([n_cells, sample_spontaneous]) # initialize the sample of spontaneous values
    spont_start_time = lsn.spont_stim_table['start'][0]
    spont_end_time = lsn.spont_stim_table['end'][0]
    spont_start_end_times = np.where(np.logical_and(spont_start_time <= times, times <= spont_end_time))[0]
    lsn_pres_len = np.ceil(tot_lsn_frames/n_tot_pres).astype(int)
    
    spont_times = spont_start_end_times[:-lsn_pres_len]
    spont_idxs = rng.choice(spont_times, sample_spontaneous) # not sure why roozbeh sliced this like this, nevertheless it gives 1000 random samples
    for s in range(sample_spontaneous):
        spont_vals[:, s] = dff[:, spont_idxs[s]:spont_idxs[s]+lsn_pres_len].mean(axis = 1) # (example length_observation was 3170)
        
    return spont_vals

def make_sample_spontanous_ns(n_cells,
                           ns,
                           times,
                           dff,
                           tot_ns_frames,
                           n_tot_ns_pres,
                           rng,
                           sample_spontaneous: int = 1000) -> np.ndarray:
    
    # define a sample of 1000 df/f values for the spontaneous activity
    spont_vals = np.zeros([n_cells, sample_spontaneous]) # initialize the sample of spontaneous values
    spont_start_time = ns.spont_stim_table['start'][0]
    spont_end_time = ns.spont_stim_table['end'][0]
    spont_start_end_times = np.where(np.logical_and(spont_start_time <= times, times <= spont_end_time))[0]
    ns_pres_len = np.ceil(tot_ns_frames/n_tot_ns_pres).astype(int)
    
    spont_times = spont_start_end_times[:-ns_pres_len]
    spont_idxs = rng.choice(spont_times, sample_spontaneous) # not sure why roozbeh sliced this like this, nevertheless it gives 1000 random samples
    for s in range(sample_spontaneous):
        spont_vals[:, s] = dff[:, spont_idxs[s]:spont_idxs[s]+ns_pres_len].mean(axis = 1) # (example length_observation was 3170)
        
    return spont_vals

def calc_lsn_vals(lsn,
                session,
                plane,
                onset_delay: float = -1,
                offset_delay: float = 2,
                trace_type: str = "events"): 
    
    # iterates through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation (i.e. mean sweep response
    # )
    traces = session.get_traces(plane, trace_type=trace_type)
    n_tot_pres = lsn.sweep_responses.shape[0]
    times = traces.indexes['time']
    dff = traces.values[lsn.is_roi_valid, :]
    n_cells = dff.shape[0]
    cell_indices = np.nonzero(lsn.is_roi_valid)[0]
    
    # iterate through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation
    lsn_vals = np.zeros([n_tot_pres, n_cells]) # initialize the sample of lsn values
    tot_lsn_frames = 0
    for cell in range(n_cells):
        tot_lsn_frames = 0
        for t in range(n_tot_pres):
            
            # version 1
            # lsn_start_time = lsn.stim_table.start[t] # 631.9549
            # lsn_end_time = lsn.stim_table.end[t] # 632.2551
            # lsn_start_end_idxs = np.where(np.logical_and(lsn_start_time <= times, times <= lsn_end_time))[0] # array([3850, 3851])
            # if len(lsn_start_end_idxs) == 0:
            #     lsn_val = 0
            # else:
            #     lsn_val = dff[cell, onset_delay + np.min(lsn_start_end_idxs): onset_delay + np.max(lsn_start_end_idxs) + offset_delay].mean(axis=0) # interval selected: (3851, 3853)
            
            
            # version 2
            # trial_start = lsn.stim_table.start[t]+onset_delay
            # trial_end = lsn.stim_table.end[t]+onset_delay+offset_delay      
            # time_mask = (times > trial_start) & (times < trial_end)
            # lsn_start_end_idxs = time_mask.nonzero()
            # if len(lsn_start_end_idxs) == 0:
            #     lsn_val = 0
            # else:
            #     lsn_val = dff[cell, time_mask].mean(axis=0) # interval selected: (3851, 3853)

            lsn_start_time = lsn.stim_table.start[t] # 631.9549
            lsn_end_time = lsn.stim_table.end[t] # 632.2551
            lsn_start_end_idxs = np.where(np.logical_and(lsn_start_time <= times, \
                times <= lsn_end_time))[0] # array([3850, 3851])
            
            if len(lsn_start_end_idxs) == 0:
                lsn_val = 0
            else:
                lsn_val = dff[cell, onset_delay + np.min(lsn_start_end_idxs): \
                    onset_delay + np.max(lsn_start_end_idxs) + offset_delay].mean(axis=0) # interval selected: (3851, 3853)
            
            lsn_vals[t, cell] = np.maximum(lsn_val, 0)
            tot_lsn_frames += len(lsn_start_end_idxs) # this is redundant and equal for every cell, fix later
    return lsn_vals, tot_lsn_frames, cell_indices  

def calc_ns12_vals(ns,
                session,
                plane,
                onset_delay: float = -1,
                offset_delay: float = 2,
                trace_type: str = "events"): 
    
    # iterates through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation (i.e. mean sweep response
    # )
    stim_table = session.get_stimulus_table("natural_images_12")
    stim_table_df = stim_table[0]
    n_images = len(stim_table_df["image"].unique())
    n_tot_pres = len(stim_table_df)
    n_repeats = n_tot_pres/n_images
   
    traces = session.get_traces(plane, trace_type=trace_type)
    # n_tot_pres = ns.sweep_responses.shape[0]
    times = traces.indexes['time']
    dff = traces.values[ns.is_roi_valid, :]
    n_cells = dff.shape[0]
    cell_indices = np.nonzero(ns.is_roi_valid)[0]
    
    # iterate through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation
    ns12_vals = np.zeros([n_tot_pres, n_cells]) # initialize the sample of lsn values
    tot_ns_frames = 0
    for cell in range(n_cells):
        tot_ns_frames = 0
        for t in range(n_tot_pres):
            
            ns_start_time = stim_table_df.iloc[t]['start']
            ns_end_time = stim_table_df.iloc[t]['end']
            ns_start_end_idxs = np.where(np.logical_and(ns_start_time <= times, \
                times <= ns_end_time))[0] # array([3850, 3851])
            
            if len(ns_start_end_idxs) == 0:
                ns_val = 0
            else:
                ns_val = dff[cell, onset_delay + np.min(ns_start_end_idxs): \
                    onset_delay + np.max(ns_start_end_idxs) + offset_delay].mean(axis=0) # interval selected: (3851, 3853)
            
            ns12_vals[t, cell] = np.maximum(ns_val, 0)
            tot_ns_frames += len(ns_start_end_idxs) # this is redundant and equal for every cell, fix later
            
    return ns12_vals, tot_ns_frames#, cell_indices, n_repeats

def calc_ns118_vals(ns,
                session,
                plane,
                onset_delay: float = -1,
                offset_delay: float = 2,
                trace_type: str = "events"): 
    
    # iterates through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation (i.e. mean sweep response
    # )
    stim_table = session.get_stimulus_table("natural_images")
    stim_table_df = stim_table[0]
    n_images = len(stim_table_df["image_index"].unique())
    n_tot_pres = len(stim_table_df)
    n_repeats = n_tot_pres/n_images
    
    traces = session.get_traces(plane, trace_type=trace_type)
    # n_tot_pres = ns.sweep_responses.shape[0]
    times = traces.indexes['time']
    dff = traces.values[ns.is_roi_valid, :]
    n_cells = dff.shape[0]
    cell_indices = np.nonzero(ns.is_roi_valid)[0]
    
    # iterate through the run and calculate the mean dff for each lsn presentation from start to the end of the presentation
    ns118_vals = np.zeros([n_tot_pres, n_cells]) # initialize the sample of lsn values
    tot_ns_frames = 0
    for cell in range(n_cells):
        tot_ns_frames = 0
        for t in range(n_tot_pres):
            
            ns_start_time = stim_table_df.iloc[t]['start']
            ns_end_time = stim_table_df.iloc[t]['end']
            ns_start_end_idxs = np.where(np.logical_and(ns_start_time <= times, \
                times <= ns_end_time))[0] # array([3850, 3851])
            
            if len(ns_start_end_idxs) == 0:
                ns_val = 0
            else:
                ns_val = dff[cell, onset_delay + np.min(ns_start_end_idxs): \
                    onset_delay + np.max(ns_start_end_idxs) + offset_delay].mean(axis=0) # interval selected: (3851, 3853)
            
            ns118_vals[t, cell] = np.maximum(ns_val, 0)
            tot_ns_frames += len(ns_start_end_idxs) # this is redundant and equal for every cell, fix later
    return ns118_vals, tot_ns_frames#, cell_indices, n_repeats

def calc_pvals_for_plane_using_zscore_one_tailed(lsn,
                         session,
                         plane,
                         lsn_vals,
                         tot_lsn_frames,rng,
                         sample_spontaneous: int = 1000,
                         trace_type: str = "events"):
   
    traces = session.get_traces(plane, trace_type=trace_type)
    
    logging.info(f"{session.get_session_id()} ({session.get_plane_depth(plane)} µm): {np.median(traces.time.diff('time')):.4f}")

    # Identify responsive trials.
    n_tot_pres = lsn.sweep_responses.shape[0]
    times = traces.indexes['time']
    dff = traces.values[lsn.is_roi_valid, :]
    n_cells = dff.shape[0]
                
    # define a sample of 1000 df/f values for the spontaneous activity
    spont_vals = make_sample_spontanous(n_cells,
                           lsn,
                           times,
                           dff,
                           tot_lsn_frames,
                           n_tot_pres,
                           rng,
                           sample_spontaneous)
    
    # calc p-vals for each lsn presentation
    p_vals = np.zeros([n_tot_pres, n_cells])
    z_vals = np.zeros([n_tot_pres, n_cells])
    
    spont_mean = np.nanmean(spont_vals)
    spont_std = np.nanstd(spont_vals)
    
    for cell in range(n_cells):
        z_vals[:, cell] = (lsn_vals[:, cell] - spont_mean)/spont_std
        p_vals [:, cell] = stats.norm.sf(abs(z_vals[:, cell])) #one-sided
 

    return p_vals, spont_vals

def calc_ns12_pvals_for_plane_using_zscore_one_tailed(ns12,
                         session,
                         plane,
                         ns12_vals,
                         tot_ns12_frames,rng,
                         sample_spontaneous: int = 1000,
                         trace_type: str = "events"):
   
    traces = session.get_traces(plane, trace_type=trace_type)
    
    logging.info(f"{session.get_session_id()} ({session.get_plane_depth(plane)} µm): {np.median(traces.time.diff('time')):.4f}")

    # Identify responsive trials.
    stim_table = session.get_stimulus_table("natural_images_12")
    stim_table_df = stim_table[0]
    n_tot_pres = len(stim_table_df)
    times = traces.indexes['time']
    dff = traces.values[ns12.is_roi_valid, :]
    n_cells = dff.shape[0]
                
    # define a sample of 1000 df/f values for the spontaneous activity
    spont_vals = make_sample_spontanous_ns(n_cells,
                           ns12,
                           times,
                           dff,
                           tot_ns12_frames,
                           n_tot_pres,
                           rng,
                           sample_spontaneous)
    
    # calc p-vals for each lsn presentation
    p_vals = np.zeros([n_tot_pres, n_cells])
    z_vals = np.zeros([n_tot_pres, n_cells])
    
    spont_mean = np.nanmean(spont_vals)
    spont_std = np.nanstd(spont_vals)
    
    for cell in range(n_cells):
        z_vals[:, cell] = (ns12_vals[:, cell] - spont_mean)/spont_std
        p_vals [:, cell] = stats.norm.sf(abs(z_vals[:, cell])) #one-sided
 

    return p_vals, spont_vals

def calc_ns118_pvals_for_plane_using_zscore_one_tailed(ns118,
                         session,
                         plane,
                         ns118_vals,
                         tot_ns118_frames,rng,
                         sample_spontaneous: int = 1000,
                         trace_type: str = "events"):
   
    traces = session.get_traces(plane, trace_type=trace_type)
    
    logging.info(f"{session.get_session_id()} ({session.get_plane_depth(plane)} µm): {np.median(traces.time.diff('time')):.4f}")

    # Identify responsive trials.
    stim_table = session.get_stimulus_table("natural_images")
    stim_table_df = stim_table[0]
    n_tot_pres = len(stim_table_df)
    times = traces.indexes['time']
    dff = traces.values[ns118.is_roi_valid, :]
    n_cells = dff.shape[0]
                
    # define a sample of 1000 df/f values for the spontaneous activity
    spont_vals = make_sample_spontanous_ns(n_cells,
                           ns118,
                           times,
                           dff,
                           tot_ns118_frames,
                           n_tot_pres,
                           rng,
                           sample_spontaneous)
    
    # calc p-vals for each lsn presentation
    p_vals = np.zeros([n_tot_pres, n_cells])
    z_vals = np.zeros([n_tot_pres, n_cells])
    
    spont_mean = np.nanmean(spont_vals)
    spont_std = np.nanstd(spont_vals)
    
    for cell in range(n_cells):
        z_vals[:, cell] = (ns118_vals[:, cell] - spont_mean)/spont_std
        p_vals [:, cell] = stats.norm.sf(abs(z_vals[:, cell])) #one-sided
 

    return p_vals, spont_vals

def get_center_indices(n):
    if n % 2 == 1:
        # If n is odd, there is one center cell
        center_row = n // 2
        center_col = n // 2
    else:
        # If n is even, there are four center cells; we'll return the top-left one
        center_row = n // 2 - 1
        center_col = n // 2 - 1
    return center_row, center_col


def rf_z_test(matrix,
              masked_z = True,
              square_mask_length: int = 7):
    
    z_value_results = np.zeros((8,14))
    
    pad_wid = (square_mask_length-1)/2
    padded_mat = np.pad(matrix, pad_width=int(pad_wid), mode='constant', constant_values= np.nan)
    
    for mat_x in range(matrix.shape[0]):
        for mat_y in range(matrix.shape[1]):
            
            if masked_z == True:
                padded_mat_x, padded_mat_y = mat_x+int(pad_wid), mat_y+int(pad_wid)            
                neighbors = padded_mat[padded_mat_x-int(pad_wid):padded_mat_x+int(pad_wid)+1, padded_mat_y-int(pad_wid):padded_mat_y+int(pad_wid)+1]

                neighbors_mean = np.nanmean(neighbors)
                neighbors_std = np.nanstd(neighbors)
                neighbors_z = (neighbors - neighbors_mean)/neighbors_std
                center_row, center_col = get_center_indices(square_mask_length)
                z_value_results[mat_x, mat_y] = neighbors_z[center_row, center_col]
            else:
                matrix_mean = np.mean(matrix)
                matrix_std = np.std(matrix)
                z_value_results = (matrix - matrix_mean)/matrix_std 
                
    zscore_result = z_value_results > 2.5
    if zscore_result.sum():
        test_outcome = True
    else:
        test_outcome = False
        
    return z_value_results, test_outcome

def cell_has_rf(weighted_avg, nstd = 3):
    val = weighted_avg.flatten()
    outcome = val >= val.mean() + nstd*val.std()
    
    if outcome.sum():
        return True
    else:
        return False

def cell_has_rf_v2(n_responsive_trials, weighted_avg, min_responsive_trials = 6, nstd = 3):
    """
    This function determines if a cell is responsive to the LSN or not
    """
    subfield_responsive_trials = n_responsive_trials
    mean_responsive_trials = subfield_responsive_trials.mean()
    std_responsive_trials = subfield_responsive_trials.std()
    max_value_responsive_trials = subfield_responsive_trials.max()
   
    subfield_all_trials = weighted_avg
    mean_all_trials = subfield_all_trials.mean()
    std_all_trials = subfield_all_trials.std()
    max_value_all_trials = subfield_all_trials.max()
   
    z_score_mat, has_rf_zscore = rf_z_test(weighted_avg);

    if max_value_responsive_trials >= min_responsive_trials and \
        max_value_responsive_trials >=  mean_responsive_trials + nstd*std_responsive_trials and \
        max_value_all_trials >=  mean_all_trials + nstd*std_all_trials and \
        has_rf_zscore:
            
        return True
    else:
        return False

def find_rf_center(n_responsive_trials_events, weighted_avg_events):
    """
    This function finds the coordinates of elements in array z(weighted_avg_off_events) that are greater than 2.5
    and checks if these coordinates match with the coordinates of the maximum value in n_responsive_trials_off_events
    or any of its neighboring coordinates within a range of ±1 in both x and y directions.

    :param n_responsive_trials_events: NumPy array of shape 8x14
    :param weighted_avg_events: NumPy array of shape 8x14
    :return: RF center
    """
    
    z_score_mat, has_rf_zscore = rf_z_test(weighted_avg_events);
    max_zscore_pixel = np.where(z_score_mat == z_score_mat.max())
    max_zscore_x = max_zscore_pixel[0][0]
    max_zscore_y = max_zscore_pixel[1][0]
    
    
    # Finding the coordinates of elements in B > 2.5
    coordinates_sig_z = np.argwhere(z_score_mat > 2.5)

    # Finding the coordinates of the maximum value in A
    max_coord_nrt = np.unravel_index(np.argmax(n_responsive_trials_events), n_responsive_trials_events.shape)

    # Generating the neighboring coordinates of the max value in A
    neighbors_a = [(max_coord_nrt[0] + dx, max_coord_nrt[1] + dy) 
                   for dx in range(-1, 2) 
                   for dy in range(-1, 2) 
                   if 0 <= max_coord_nrt[0] + dx < n_responsive_trials_events.shape[0] 
                   and 0 <= max_coord_nrt[1] + dy < n_responsive_trials_events.shape[1]]

    # Checking if any coordinates in B match with the max or its neighbors in A
    matching_coords = [coord for coord in coordinates_sig_z if tuple(coord) in neighbors_a]

    if matching_coords:
        rf_center_x, rf_center_y = matching_coords[0][0], matching_coords[0][1]
    else:
        rf_center_x, rf_center_y = max_zscore_x, max_zscore_y
    
    gauss_input_init = z_score_mat > 2.5
    gauss_input = np.zeros_like(gauss_input_init)   
    # Define the range for rows and columns
    row_range = range(max(0, rf_center_x - 2), min(n_responsive_trials_events.shape[0], rf_center_x + 3))
    col_range = range(max(0, rf_center_y - 2), min(n_responsive_trials_events.shape[1], rf_center_y + 3))

    # Copy the values from the original array
    for i in row_range:
        for j in col_range:
            gauss_input[i, j] = gauss_input_init[i, j]
        
    return rf_center_x, rf_center_y, gauss_input

def assign_angle(data, center_x, center_y):
    rows, cols = data.shape
    # Check if the swapped center coordinates are within the array bounds
    if center_y < 0 or center_y >= cols or center_x < 0 or center_x >= rows:
        return None

    # Check the conditions with swapped coordinates and assign angles in degrees
    if (center_y + 1 < cols and data[center_x, center_y + 1] != 0) or (center_y - 1 >= 0 and data[center_x, center_y - 1] != 0):
        angle_degrees = 0
    elif (center_x + 1 < rows and data[center_x + 1, center_y] != 0)or (center_x - 1 >= 0 and data[center_x - 1, center_y] != 0):
        angle_degrees = 90
    elif (center_y + 1 < cols and center_x + 1 < rows and data[center_x + 1, center_y + 1] != 0) or (center_y - 1 >= 0 and center_x - 1 >= 0 and data[center_x - 1, center_y - 1] != 0):
        angle_degrees = 135
    elif (center_y - 1 >= 0 and center_x + 1 < rows and data[center_x + 1, center_y - 1] != 0) or (center_y + 1 < cols and center_x - 1 >= 0 and data[center_x - 1, center_y + 1] != 0):
        angle_degrees = 45
    else:
        return None  # No conditions met

    # # Convert angle from degrees to radians
    # angle_radians = angle_degrees * math.pi / 180
    return angle_degrees

def find_rf_center_v2(n_responsive_trials_events, weighted_avg_events):
    """_summary_

    Args:
        n_responsive_trials_events (_type_): _description_
        weighted_avg_events (_type_): _description_

        This function finds the RF center based on the maximum of z_score_mat, remove n_responsive_trials_events later.
    Returns:
        _type_: _description_
    """
    
    z_score_mat, has_rf_zscore = rf_z_test(weighted_avg_events);
    max_zscore_pixel = np.where(z_score_mat == z_score_mat.max())
    rf_center_x = max_zscore_pixel[0][0]
    rf_center_y = max_zscore_pixel[1][0]

    gauss_input_init = z_score_mat > 2.5
    gauss_input = np.zeros_like(gauss_input_init)   
    # Define the range for rows and columns
    row_range = range(max(0, rf_center_x - 1), min(weighted_avg_events.shape[0], rf_center_x + 2))
    col_range = range(max(0, rf_center_y - 1), min(weighted_avg_events.shape[1], rf_center_y + 2))

    # Copy the values from the original array
    for i in row_range:
        for j in col_range:
            gauss_input[i, j] = gauss_input_init[i, j]
        
    return rf_center_x, rf_center_y, gauss_input
  
class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, weights):
        
        super().__init__()
        # make weights torch parameters
        self.weights = nn.Parameter(weights)   
        
        
    def forward(self, X, f_type):
        """Implement 2d Gaussian function.
        """
        # height, center_x, center_y, width = self.weights
        gaussian_on_plan = lambda p: torch.ravel(func(*p, f_type=f_type)(*X))
        return gaussian_on_plan(self.weights) # 


def func(center_x, center_y, height, width_x, width_y, angle, f_type):
    """
    Returns a function based on f_type. For Gaussian types, includes rotation.
    Angle is in radians.
    """
    theta = angle  # angle in radians
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    if f_type == 1:
        # Rotated Gaussian
        return lambda x, y: height * torch.exp(-0.5 * (
            ((cos_theta * (x - center_x) + sin_theta * (y - center_y)) ** 2 / torch.abs(width_x)) +
            ((sin_theta * (x - center_x) - cos_theta * (y - center_y)) ** 2 / torch.abs(width_y))
        ))

    elif f_type == 2:
        # Non-Gaussian type (original implementation)
        return lambda x, y: height / (0.1 + (center_x - x) ** 2 + (center_y - y) ** 2)

    elif f_type == 3:
        # Non-Gaussian type (original implementation)
        return lambda x, y: height / (0.1 + torch.abs(center_x - x) + torch.abs(center_y - y))

    elif f_type == 4:
        # Rotated Gaussian with different x and y widths
        return lambda x, y: height * torch.exp(-0.5 * (
            ((cos_theta * (x - center_x) + sin_theta * (y - center_y)) ** 2 / torch.abs(width_x)) +
            ((sin_theta * (x - center_x) - cos_theta * (y - center_y)) ** 2 / torch.abs(width_y))
        ))

    # Add more cases if necessary
    else:
        raise ValueError("Unsupported function type")



def training_loop(model, 
                  y, 
                  optimizer, 
                  balance_weight=1.0,
                  norm_weight = 0.0002,
                  f_type=1, 
                  n_training=200, 
                  dim = [8, 14]):
    "Training loop for torch model."
    losses = []
    params = []
    x = torch.from_numpy(np.indices(dim))
    for i in range(n_training):
        preds = model(x, f_type=f_type)
        # first implentation
        #s = max(torch.abs(preds - y))
        #loss = F.mse_loss(s, torch.tensor([0.0])) + balance_weight* F.mse_loss(preds, y).sqrt()
        
        # second implentation
        diff = torch.abs(preds - y)
        m = nn.MaxPool1d((x[0].shape[0]*x[0].shape[1]))
        
        # Third implentation
        loss = torch.sqrt(F.mse_loss(preds, y)) + norm_weight * model.weights[3:].norm() + balance_weight * torch.squeeze(m(diff[None, :]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(torch.sqrt(F.mse_loss(preds, y)).detach().numpy())
        params.append(list(model.parameters())[0][:].detach().numpy())
    return losses, params

def centroid(data, 
            balance_weight=1.0,
            norm_weight = 0.08,
             lr=0.1, 
             weight_decay=0.03, 
             f_type=4,
             n_training=1000, 
             initialization = 'default',
             initial_weight=[1,1,1,1,1,0],
            return_loss=False):
    if f_type==5:
        return discrete_width(data, n_grid = 7)

    y = torch.ravel(torch.from_numpy(data)).float()
    indcies = np.unravel_index(np.argmax(data, axis=None), data.shape)
    max_x, max_y, height = indcies[0], indcies[1], data.max()
    if initialization == 'default':
        weights = torch.tensor([float(max_x), float(max_y), float(height), 1, 1, 0])  # 0 is the initial angle

    else:
        weights = torch.tensor(initial_weight)
    
    if f_type==6:
        shift = 2
        if np.abs(max_x-4) <4-shift+1 and np.abs(max_y-7) <7-shift+1:   
            data_crop = data[max_x-shift:max_x+shift, max_y-shift:max_y+shift]
        else:
            shift = 0
            data_crop = data
        center_x_crop, center_y_crop, height_crop, width_x_crop, width_y_crop = centroid(data_crop, 
                balance_weight=balance_weight,
                norm_weight = norm_weight,
                 lr=lr, 
                 weight_decay=weight_decay, 
                 f_type=5,
                 n_training=n_training, 
                 initialization = initialization,
                 initial_weight=initial_weight,
                return_loss=False)

        return max_x - shift+center_x_crop, max_y - shift+center_y_crop, height_crop, width_x_crop, width_y_crop
    m = Model(weights)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=weight_decay)
    losses, training_params = training_loop(m, 
                                            y, 
                                            opt, 
                                            balance_weight=balance_weight, 
                                            f_type=f_type, 
                                            n_training=n_training,
                                           dim = data.shape)
    center_x, center_y, height, width_x, width_y, angle = m.weights  
    params =  (center_x.detach().numpy(), center_y.detach().numpy(), height.detach().numpy(), 
               np.abs(width_x.detach().numpy()), np.abs(width_y.detach().numpy()), angle.detach().numpy())
    
    print(return_loss)
    if return_loss:
        return params, losses, training_params, m
    else:
        return params

def discrete_width(data, n_grid=7):
    x, y = np.indices(data.shape)
    best_fit_val = np.inf
    best_width_x = 0
    best_width_y = 0
    best_angle = 0

    for width_x in np.arange(1, (n_grid + 1) * .5, .5):
        for width_y in np.arange(1, (n_grid + 1) * .5, .5):
            for angle in np.linspace(0, np.pi, n_grid):  # Searching over angles from 0 to pi
                indices = np.unravel_index(np.argmax(data, axis=None), data.shape)
                center_x, center_y, height = float(indices[0]), float(indices[1]), float(data.max())

                rotated_gaussian = lambda x, y: gaussian(x, y, center_x, center_y, width_x, width_y, height, angle)

                if np.sum((rotated_gaussian(x, y) - data) ** 2) < best_fit_val:
                    best_fit_val = np.sum((rotated_gaussian(x, y) - data) ** 2)
                    best_width_x = width_x
                    best_width_y = width_y
                    best_angle = angle

    return center_x, center_y, height, best_width_x, best_width_y, best_angle

def gaussian(x, y, x_center, y_center, x_width, y_width, height, angle):
    x_prime = np.cos(angle) * (x - x_center) + np.sin(angle) * (y - y_center)
    y_prime = -np.sin(angle) * (x - x_center) + np.cos(angle) * (y - y_center)

    x_term = (x_prime ** 2) / (2 * x_width ** 2)
    y_term = (y_prime ** 2) / (2 * y_width ** 2)

    return height * np.exp(-(x_term + y_term))
def compute_lsn_design_matrix(lsn):
    stim_pixels = lsn.frame_images[lsn.stim_table.frame.values].reshape((-1, lsn.n_pixels))
    design_matrix_on = np.where(stim_pixels == lsn.pixel_on, True, False)
    design_matrix_off = np.where(stim_pixels == lsn.pixel_off, True, False)
    design_matrix = np.concatenate((design_matrix_on, design_matrix_off), axis=1)
    design_matrix = design_matrix.T # shape (2*n_pixels, n_sweeps)

    design_matrix_on2 = design_matrix_on.astype(int)
    design_matrix_on2[design_matrix_on2 == 0] = 127
    design_matrix_on2[design_matrix_on2 == 1] = 255

    design_matrix_off2 = design_matrix_off.astype(int)
    design_matrix_off2[design_matrix_off2 == 0] = 127
    design_matrix_off2[design_matrix_off2 == 1] = 0

    new_design_matrix = design_matrix_on2 + design_matrix_off2

    new_design_matrix = new_design_matrix.T
    frames = new_design_matrix.reshape(8, 14, 1705)
    return new_design_matrix, frames
   

def calc_lsn_p_vals_for_col_vol_plane(session: OPhysSession,
                                      plane: int,
                                      sample_spontaneous: int = 1000,
                                      onset_delay: float = -1,
                                      offset_delay: float = 2,
                                      trace_type: str = "events"):

    all_pvals = []
    all_lsn_vals = []
        
    lsn = LocallySparseNoise(session, plane, trace_type = trace_type)
    lsn_vals, tot_lsn_frames, cell_indices = calc_lsn_vals(lsn,
                    session,
                    plane,
                    onset_delay,
                    offset_delay,
                    trace_type=trace_type)
    all_lsn_vals.append(lsn_vals)
    
    if tot_lsn_frames > 0:
        p_vals, _ = calc_pvals_for_plane_using_zscore_one_tailed(lsn,
                            session,
                            plane,
                            lsn_vals,
                            tot_lsn_frames,
                            rng,
                            sample_spontaneous,
                            trace_type=trace_type
                            )

        all_pvals.append(p_vals)
    
    all_lsn_vals = np.array(all_lsn_vals).squeeze()
    all_pvals = np.array(all_pvals).squeeze()
            
    return all_lsn_vals, all_pvals, cell_indices

def calc_ns12_p_vals_for_col_vol_plane(session: OPhysSession,
                                      plane: int,
                                      sample_spontaneous: int = 1000,
                                      onset_delay: float = -1,
                                      offset_delay: float = 2,
                                      trace_type: str = "events"):

    all_ns_pvals = []
    all_ns_vals = []
        
    ns = NaturalScenes(session, plane, trace_type = trace_type)
    stim_table = session.get_stimulus_table("natural_images_12")
    stim_table_df = stim_table[0]
    n_images = len(stim_table_df["image"].unique())
    
    ns_vals, tot_lsn_frames = calc_ns12_vals(ns,
                    session,
                    plane,
                    onset_delay,
                    offset_delay,
                    trace_type=trace_type)
    all_ns_vals.append(ns_vals)
    
    if tot_lsn_frames > 0:
        ns_p_vals, _ = calc_ns12_pvals_for_plane_using_zscore_one_tailed(ns,
                            session,
                            plane,
                            ns_vals,
                            tot_lsn_frames,
                            rng,
                            sample_spontaneous,
                            trace_type=trace_type
                            )

        all_ns_pvals.append(ns_p_vals)
    
    all_ns_vals = np.array(all_ns_vals).squeeze()
    all_ns_pvals = np.array(all_ns_pvals).squeeze()

    ns_means = np.array([ns_vals[stim_table_df[stim_table_df["image"] == i].index].mean(axis=0) for i in range(n_images)])
    all_pref_images = np.argmax(ns_means, axis=0)
                    
    return all_ns_vals, all_ns_pvals, all_pref_images

def calc_ns118_p_vals_for_col_vol_plane(session: OPhysSession,
                                      plane: int,
                                      sample_spontaneous: int = 1000,
                                      onset_delay: float = -1,
                                      offset_delay: float = 2,
                                      trace_type: str = "events"):

    all_ns_pvals = []
    all_ns_vals = []
        
    ns = NaturalScenes(session, plane, trace_type = trace_type)
    stim_table = session.get_stimulus_table("natural_images")
    stim_table_df = stim_table[0]
    n_images = len(stim_table_df["image_index"].unique())
    
    ns_vals, tot_lsn_frames = calc_ns118_vals(ns,
                    session,
                    plane,
                    onset_delay,
                    offset_delay,
                    trace_type=trace_type)
    all_ns_vals.append(ns_vals)
    
    if tot_lsn_frames > 0:
        ns_p_vals, _ = calc_ns118_pvals_for_plane_using_zscore_one_tailed(ns,
                            session,
                            plane,
                            ns_vals,
                            tot_lsn_frames,
                            rng,
                            sample_spontaneous,
                            trace_type=trace_type
                            )

        all_ns_pvals.append(ns_p_vals)
    
    all_ns_vals = np.array(all_ns_vals).squeeze()
    all_ns_pvals = np.array(all_ns_pvals).squeeze()

    ns_means = np.array([ns_vals[stim_table_df[stim_table_df["image_index"] == i].index].mean(axis=0) for i in range(n_images)])
    all_pref_images = np.argmax(ns_means, axis=0)
                    
    return all_ns_vals, all_ns_pvals, all_pref_images
            
            
def get_plane_lsn_constants(session: OPhysSession,
                            trace_type: str = "events"):
    planes = session.get_planes()
    if len(planes) == 0:
        raise ValueError('No planes found')
    plane = planes[0] # assumes first plane has shared constants as other ones  
    lsn = LocallySparseNoise(session, plane, trace_type = trace_type)
    design_matrix = lsn.design_matrix.astype(int)
    return design_matrix, lsn.trial_template, lsn.frame_images


def compute_rf_ns_metrics_for_col_vol_plane(client: OPhysClient, 
                        mouse_id: str, 
                        col_vol_id: str,
                        plane: int,
                        sample_spontaneous: int = 1000,
                        onset_delay: float = -1,
                        offset_delay: float = 2,
                        response_thresh_alpha: float = 0.05,
                        nstd: float = 3,
                        trace_type: str = "events") -> dict:

    session = client.load_ophys_session(f"{mouse_id}_{col_vol_id}")

    if session is None:
        raise ValueError('Session not found')

    all_ns12_vals_in_colvol_plane, all_ns12_pvals_in_colvol_plane, ns12_pref_images  = calc_ns12_p_vals_for_col_vol_plane(session,
                                        plane,
                                        sample_spontaneous,
                                        onset_delay,
                                        offset_delay,
                                        trace_type=trace_type)
    
    all_ns118_vals_in_colvol_plane, all_ns118_pvals_in_colvol_plane, ns118_pref_images  = calc_ns118_p_vals_for_col_vol_plane(session,
                                    plane,
                                    sample_spontaneous,
                                    onset_delay,
                                    offset_delay,
                                    trace_type=trace_type)
        
    all_lsn_vals_in_colvol_plane, all_pvals_in_colvol_plane, cell_indices  = calc_lsn_p_vals_for_col_vol_plane(session,
                                        plane,
                                        sample_spontaneous,
                                        onset_delay,
                                        offset_delay,
                                        trace_type=trace_type)
    
    design_matrix, trial_template, frame_images = get_plane_lsn_constants(session, 
                                                                          trace_type=trace_type)
    
    all_x = []
    all_y = []
    all_depths = []
    all_planes = []
    all_columns = []
    all_volumes = []
    all2p3ps = []

    
    lsn = LocallySparseNoise(session, plane, trace_type = trace_type)
    ns = NaturalScenes(session, plane, trace_type = trace_type)

    for icell in range(lsn.n_rois_valid):
        roi_mask = session.get_roi_image_mask(plane, icell)
        y1, x1 = np.mean(np.where(roi_mask), axis=1, dtype=int)
        depth = session.get_plane_depth(plane)
        all_x.append(x1)
        all_y.append(y1)
        all_depths.append(depth)
        all_planes.append(plane)
        all_columns.append(session.column_id)
        all_volumes.append(session.volume_id)
        all2p3ps.append(int(session.scope_type[0]))
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)    
    all_depths = np.array(all_depths)
    all_planes = np.array(all_planes)
    all_columns = np.array(all_columns)
    all_volumes = np.array(all_volumes)
    all2p3ps = np.array(all2p3ps)

    rf_metrics = {
        "mouse_id": mouse_id,
        "column": session.column_id,            
        "volume": session.volume_id,
        "col_vol": col_vol_id,
        "plane": plane,
        "data": {},
        "trace_type": trace_type
    }

    if lsn.n_rois_valid > 1:
        n_valid_cells_in_colvol_plane = lsn.n_rois_valid # all_pvals_in_colvol_plane.shape[1]
        n_trials = all_pvals_in_colvol_plane.shape[0]
        
        # initialization
        
        valid_cell_index = np.zeros(n_valid_cells_in_colvol_plane, dtype = int) 
        on_center_x = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_center_y = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_center_h = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_angle = np.zeros(n_valid_cells_in_colvol_plane, dtype = float) 
        off_center_x = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_center_y = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_center_h = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_angle = np.zeros(n_valid_cells_in_colvol_plane, dtype = float) 
        on_angle_degree = np.zeros(n_valid_cells_in_colvol_plane, dtype = float) 
        off_angle_degree = np.zeros(n_valid_cells_in_colvol_plane, dtype = float) 
        on_center_wx = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_center_wy = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_center_wx = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_center_wy = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_area = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_area = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        on_averaged_response_at_receptive_field = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        off_averaged_response_at_receptive_field = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        percentage_res_trial_4_locally_sparse_noise = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        frac_res_trial_4_locally_sparse_noise = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        frac_res_to_on = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        frac_res_to_off = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        frac_res_to_ns12 = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        frac_res_to_ns118 = np.zeros(n_valid_cells_in_colvol_plane, dtype = float)
        
        number_of_pixels = int(design_matrix.shape[0]/2) # 112
        number_of_tot_pixels = int(design_matrix.shape[0]/2)*2 # 224
        total_on_off_trials = design_matrix.dot(np.ones(n_trials))

        s1, s2 = 8, 14
        
        lsn_values = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        p_values = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        is_trial_sig = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        total_responsive_trials_all_pixels = np.zeros(n_valid_cells_in_colvol_plane)
        n_responsive_trials = np.zeros((n_valid_cells_in_colvol_plane, number_of_tot_pixels), dtype = int)
        weighted_avg = np.zeros((n_valid_cells_in_colvol_plane, number_of_tot_pixels))
        weighted_avg_only_resp_trials = np.zeros((n_valid_cells_in_colvol_plane, number_of_tot_pixels))
        is_responsive = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_mean_std = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_chi2 = np.zeros(n_valid_cells_in_colvol_plane)
        chi2_mat_thresholded = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))

        n_responsive_trials_on = np.zeros((n_valid_cells_in_colvol_plane, s1, s2), dtype = int)
        weighted_avg_on = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        weighted_avg_only_resp_trials_on = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        max_n_responsive_trials_on = np.zeros(n_valid_cells_in_colvol_plane, dtype = int)
        is_responsive_to_on = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_mean_std_on = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_v2_on = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        z_score_mat_on = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        has_rf_zscore_on = np.zeros(n_valid_cells_in_colvol_plane)
        sig_on_frames = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        max_wavg_on_frames = np.zeros((n_valid_cells_in_colvol_plane, n_trials))    
        has_on_rf = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)

        n_responsive_trials_off = np.zeros((n_valid_cells_in_colvol_plane, s1, s2), dtype = int)
        weighted_avg_off = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        weighted_avg_only_resp_trials_off = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        max_n_responsive_trials_off = np.zeros(n_valid_cells_in_colvol_plane, dtype = int)
        is_responsive_to_off = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_mean_std_off = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        has_rf_v2_off = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)
        z_score_mat_off = np.zeros((n_valid_cells_in_colvol_plane, s1, s2))
        has_rf_zscore_off = np.zeros(n_valid_cells_in_colvol_plane)
        sig_off_frames = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        max_wavg_off_frames = np.zeros((n_valid_cells_in_colvol_plane, n_trials))
        has_off_rf = np.zeros(n_valid_cells_in_colvol_plane, dtype = bool)

        chi2_mat = chi_square_RFs(all_lsn_vals_in_colvol_plane, trial_template)
        mu_spont = np.mean(all_lsn_vals_in_colvol_plane, 0)
        max_spont = np.max(all_lsn_vals_in_colvol_plane, 0)
        min_spont = np.min(all_lsn_vals_in_colvol_plane, 0)
        gauss_w_coeff = 7
        
        stim_table_ns118 = session.get_stimulus_table("natural_images")
        stim_table_ns118_df = stim_table_ns118[0]
        n_images_ns118 = len(stim_table_ns118_df["image_index"].unique())
        n_tot_pres_ns118 = len(stim_table_ns118_df)
        n_repeats_ns118 = n_tot_pres_ns118/n_images_ns118
    
        stim_table_ns12 = session.get_stimulus_table("natural_images_12")
        stim_table_ns12_df = stim_table_ns12[0]
        n_images_ns12 = len(stim_table_ns12_df["image"].unique())
        n_tot_pres_ns12 = len(stim_table_ns12_df)
        n_repeats_ns12 = n_tot_pres_ns12/n_images_ns12
 
        ns12_pvals_means = np.array([all_ns12_pvals_in_colvol_plane[stim_table_ns12_df[stim_table_ns12_df["image"] == ns12_pref_images[idx]].index, idx] for \
            idx in range(n_valid_cells_in_colvol_plane)])
    
        ns118_pvals_means = np.array([all_ns118_pvals_in_colvol_plane[stim_table_ns118_df[stim_table_ns118_df["image_index"] == ns118_pref_images[idx]].index, idx] for \
            idx in range(n_valid_cells_in_colvol_plane)])
                
  
        # compute rf for each volume
        for cell in range(n_valid_cells_in_colvol_plane):

            frac_res_to_ns12[cell] = (ns12_pvals_means[cell,:]<response_thresh_alpha).sum()/n_repeats_ns12
            frac_res_to_ns118[cell] = (ns118_pvals_means[cell,:]<response_thresh_alpha).sum()/n_repeats_ns118

            # rf metrics
            valid_cell_index[cell] = cell
            
            lsn_values[cell, :] = all_lsn_vals_in_colvol_plane[:, cell] 
            p_values[cell, :] = all_pvals_in_colvol_plane[:, cell]
            is_trial_sig[cell, :] = all_pvals_in_colvol_plane[:, cell] < response_thresh_alpha

            total_responsive_trials_all_pixels[cell]= is_trial_sig[cell, :].sum() 
        
            n_responsive_trials[cell, :] = design_matrix.dot(is_trial_sig[cell, :]) # just average
            percentage_res_trial_4_locally_sparse_noise[cell] = total_responsive_trials_all_pixels[cell]*100/is_trial_sig.shape[1]
            frac_res_trial_4_locally_sparse_noise[cell] = total_responsive_trials_all_pixels[cell]/is_trial_sig.shape[1]
            
            weighted_avg[cell, :] = design_matrix.dot(all_lsn_vals_in_colvol_plane[:, cell]) # weighted average (based on responsiveness score)
            
            only_resp_trials_design_matrix = design_matrix[:, is_trial_sig[cell, :].astype(bool)]
            lsn_vals_only_resp_trials = all_lsn_vals_in_colvol_plane[is_trial_sig[cell, :].astype(bool), cell]
            weighted_avg_only_resp_trials[cell, :] = only_resp_trials_design_matrix.dot(lsn_vals_only_resp_trials) # weighted average only based on responsive trials (based on responsiveness score)

            is_responsive[cell] = is_cell_responsive(n_responsive_trials[cell, :], weighted_avg[cell, :], min_responsive_trials = 8, nstd = 3);
            has_rf_mean_std[cell] = cell_has_rf(weighted_avg[cell, :], nstd = nstd);
            chi2_mat_thresholded[cell, :, :] = chi2_mat[cell, :, :] < 0.05
            
            has_rf_chi2[cell] = chi2_mat_thresholded[cell, :, :].sum().astype(bool)

            # on analysis
            # total_on_trials = total_on_off_trials[:number_of_pixels].sum()
            # n_resp_on_trials = n_responsive_trials[cell, :number_of_pixels].sum()
            # frac_res_to_on[cell] = n_resp_on_trials / total_on_trials

            n_responsive_trials_on[cell, :, :] = n_responsive_trials[cell, :number_of_pixels].reshape(s1, s2)
            total_on_trials = total_on_off_trials[:number_of_pixels].reshape(s1, s2)
            frac_res_to_on[cell] = (n_responsive_trials_on[cell, :, :] / total_on_trials).max()
            
            on_frame_idxs = np.multiply(design_matrix[n_responsive_trials_on[cell, :, :].argmax()], is_trial_sig[cell, :].T).nonzero()[0]
            sig_on_frames[cell, on_frame_idxs] = 1
            
            on_wavg_frame_idxs = np.multiply(design_matrix[weighted_avg_on[cell, :, :].argmax()], is_trial_sig[cell, :].T).nonzero()[0]
            max_wavg_on_frames[cell, on_wavg_frame_idxs] = 1

            weighted_avg_on[cell, :, :] = weighted_avg[cell, :number_of_pixels].reshape(s1, s2)
            weighted_avg_only_resp_trials_on[cell, :, :] = weighted_avg_only_resp_trials[cell, :number_of_pixels].reshape(s1, s2)
            max_n_responsive_trials_on[cell] = n_responsive_trials_on[cell, :].max()
            is_responsive_to_on[cell] = is_cell_responsive(n_responsive_trials_on[cell, :], weighted_avg_on[cell, :], min_responsive_trials = 8, nstd = 3);
            has_rf_mean_std_on [cell] = cell_has_rf(weighted_avg_on[cell, :], nstd = nstd);
            has_rf_v2_on[cell] = cell_has_rf_v2(n_responsive_trials_on[cell, :], weighted_avg_on[cell, :], min_responsive_trials = 7, nstd = 3);
            z_score_mat_on[cell, :, :], has_rf_zscore_on[cell] = rf_z_test(weighted_avg_on[cell, :]);
            on_averaged_response_at_receptive_field[cell] = weighted_avg_on[cell, :, :].max()
            
            # gauss_input = z_score_mat_on[cell, :, :] > 2.5
            # gauss_input_argmax = np.where(gauss_input == gauss_input.max())
            # x_initial = gauss_input_argmax[0]
            # y_initial = gauss_input_argmax[1]
            x_initial, y_initial, gauss_input = find_rf_center_v2(n_responsive_trials_on[cell, :], weighted_avg_on[cell, :])
            h_initial = gauss_input.max()
            on_angle_deg = assign_angle(gauss_input, x_initial, y_initial)

            on_params = centroid(gauss_input,
                            initial_weight=[x_initial,y_initial,h_initial,1,1]) #center_x, center_y, height, width_x, width_y)
        
            on_center_x[cell] = on_params[1]
            on_center_y[cell] = on_params[0]
            on_center_h[cell] = on_params[2]
            on_center_wx[cell] = on_params[4]*gauss_w_coeff
            on_center_wy[cell] = on_params[3]*gauss_w_coeff
            on_angle[cell] = -1*on_params[5]
            on_angle_degree[cell] = on_angle_deg           
            on_area[cell] = np.pi * on_center_wx[cell] * on_center_wy[cell]
            has_on_rf [cell] = (on_center_wx[cell] < 5) and (on_center_wy[cell] < 5) and h_initial > 2.5
            
            # FIX ME: if wx or wy is > 4 means it did not converge. Find solution.

            # off analysis
            # total_off_trials = total_on_off_trials[number_of_pixels:].sum()
            # n_resp_off_trials = n_responsive_trials[cell, number_of_pixels:].sum()
            # frac_res_to_off[cell] = n_resp_off_trials / total_off_trials
            
            n_responsive_trials_off[cell, :, :] = n_responsive_trials[cell, number_of_pixels:].reshape(s1, s2)
            total_off_trials = total_on_off_trials[number_of_pixels:].reshape(s1, s2)
            frac_res_to_off[cell] = (n_responsive_trials_off[cell, :, :] / total_off_trials).max()
            
            off_frame_idxs = np.multiply(design_matrix[n_responsive_trials_off[cell, :, :].argmax()+112], is_trial_sig[cell, :].T).nonzero()[0]
            sig_off_frames [cell, off_frame_idxs] = 1

            off_wavg_frame_idxs = np.multiply(design_matrix[weighted_avg_on[cell, :, :].argmax()+112], is_trial_sig[cell, :].T).nonzero()[0]
            max_wavg_off_frames[cell, off_wavg_frame_idxs] = 1        
            
            weighted_avg_off[cell, :, :] = weighted_avg[cell, number_of_pixels:].reshape(s1, s2)
            weighted_avg_only_resp_trials_off[cell, :, :] = weighted_avg_only_resp_trials[cell, number_of_pixels:].reshape(s1, s2)  
            max_n_responsive_trials_off [cell] = n_responsive_trials_off[cell, :].max()
            is_responsive_to_off[cell] = is_cell_responsive(n_responsive_trials_off[cell, :], weighted_avg_off[cell, :], min_responsive_trials = 8, nstd = 3);
            has_rf_mean_std_off[cell] = cell_has_rf(weighted_avg_off[cell, :], nstd = nstd);
            has_rf_v2_off[cell] = cell_has_rf_v2(n_responsive_trials_off[cell, :], weighted_avg_off[cell, :], min_responsive_trials = 7, nstd = 3);
            z_score_mat_off[cell, :, :], has_rf_zscore_off[cell] = rf_z_test(weighted_avg_off[cell, :]);
            
            off_averaged_response_at_receptive_field[cell] = weighted_avg_off[cell, :, :].max()
            
            
            # gauss_input = z_score_mat_off[cell, :, :] > 2.5
            # gauss_input_argmax = np.where(gauss_input == gauss_input.max())
            # x_initial = gauss_input_argmax[0]
            # y_initial = gauss_input_argmax[1]
            x_initial, y_initial, gauss_input = find_rf_center_v2(n_responsive_trials_off[cell, :], weighted_avg_off[cell, :])
            h_initial = gauss_input.max()
            off_angle_deg = assign_angle(gauss_input, x_initial, y_initial)

            off_params = centroid(gauss_input,
                            initial_weight=[x_initial,y_initial,h_initial,1,1]) #center_x, center_y, height, width_x, width_y)
            off_center_x[cell] = off_params[1]
            off_center_y[cell] = off_params[0]
            off_center_h[cell] = off_params[2]
            off_center_wx[cell] = off_params[4]*gauss_w_coeff
            off_center_wy[cell] = off_params[3]*gauss_w_coeff
            off_angle[cell] = -1*off_params[5]
            off_angle_degree[cell] = off_angle_deg
           
            off_area[cell] = np.pi * off_center_wx[cell] * off_center_wy[cell]
            has_on_rf [cell] = (off_center_wx[cell] < 5) and (off_center_wy[cell] < 5) and h_initial > 2.5 
            # FIX ME: if wx or wy is > 4 means it did not converge. Find solution
        
        rf_metrics["data"] = {
                "sig_on_frames":sig_on_frames,
                "sig_off_frames":sig_off_frames,
                "max_wavg_on_frames":max_wavg_on_frames,
                "max_wavg_off_frames":max_wavg_off_frames,        
                "frame_images":frame_images,
                "design_matrix":lsn.design_matrix,
                "lsn_values":lsn_values,
                "p_values":p_values,            
                "mu_spont":mu_spont,
                "max_spont":max_spont,
                "min_spont":min_spont,

                "is_trial_sig":is_trial_sig,
                "total_responsive_trials_all_pixels":total_responsive_trials_all_pixels,
                "n_responsive_trials":n_responsive_trials,
                "weighted_avg":weighted_avg,
                "weighted_avg_only_resp_trials":weighted_avg_only_resp_trials,    
                "is_responsive":is_responsive,
                "has_rf_mean_std":has_rf_mean_std,
                "chi2_mat":chi2_mat,
                "chi2_mat_thresholded":chi2_mat_thresholded,
                "has_rf_chi2":has_rf_chi2,

                "n_responsive_trials_on":n_responsive_trials_on,
                "weighted_avg_on":weighted_avg_on,    
                "weighted_avg_only_resp_trials_on":weighted_avg_only_resp_trials_on,    
                "max_n_responsive_trials_on":max_n_responsive_trials_on,
                "is_responsive_to_on":is_responsive_to_on,
                "has_rf_mean_std_on":has_rf_mean_std_on,
                "has_rf_v2_on":has_rf_v2_on,
                "z_score_mat_on":z_score_mat_on,
                "has_rf_zscore_on":has_rf_zscore_on,

                "n_responsive_trials_off":n_responsive_trials_off,
                "weighted_avg_off":weighted_avg_off,
                "weighted_avg_only_resp_trials_off":weighted_avg_only_resp_trials_off,       
                "max_n_responsive_trials_off":max_n_responsive_trials_off,
                "is_responsive_to_off":is_responsive_to_off,
                "has_rf_mean_std_off":has_rf_mean_std_off,
                "has_rf_v2_off":has_rf_v2_off,
                "z_score_mat_off":z_score_mat_off,    
                "has_rf_zscore_off":has_rf_zscore_off,
                
                "valid_cell_index": valid_cell_index,
                "cell_index": cell_indices,
                "x": all_x,
                "y": all_y,
                "z": all_depths,
                "2p3p": all2p3ps,
                "on_score": max_n_responsive_trials_on,
                "off_score": max_n_responsive_trials_off,
                "on_center_x": on_center_x,
                "on_center_y": on_center_y,
                "on_center_h": on_center_h,
                "off_center_x": off_center_x,
                "off_center_y": off_center_y,
                "off_center_h": off_center_h,
                "on_center_wx": on_center_wx,
                "on_center_wy": on_center_wy,
                "off_center_wx": off_center_wx,
                "off_center_wy": off_center_wy,
                "on_angle": on_angle,
                "off_angle": off_angle,
                "on_angle_degree":on_angle_degree,
                "off_angle_degree":off_angle_degree,  
                "on_area": on_area, 
                "off_area": off_area,
                "has_on_rf": has_on_rf, 
                "has_off_rf": has_off_rf,            
                "on_averaged_response_at_receptive_field": on_averaged_response_at_receptive_field,
                "off_averaged_response_at_receptive_field": off_averaged_response_at_receptive_field,
                "percentage_res_trial_4_locally_sparse_noise": percentage_res_trial_4_locally_sparse_noise,
                "frac_res_trial_4_locally_sparse_noise":frac_res_trial_4_locally_sparse_noise,
                "frac_res_to_on": frac_res_to_on,
                "frac_res_to_off": frac_res_to_off,
                "frac_res_to_ns12": frac_res_to_ns12,
                "frac_res_to_ns118": frac_res_to_ns118
                }
    return rf_metrics
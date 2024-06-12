# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:16:53 2018

@author: danielm
"""
import numpy as np

def chi_square_RFs(responses,LSN_template,num_shuffles=1000):
    
    #responses: numpy array with shape (num_trials,num_cells)
    #LSN_template: numpy array with shape (num_trials,num_y,num_x)
    
    num_trials = np.shape(responses)[0]
    num_cells = np.shape(responses)[1]        
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]

    assert num_trials == np.shape(LSN_template)[0]

    #remove negative responses:
    responses = np.where(responses>=0.0,responses,0.0)

    #define the masks for exclusion regions we'll calculate the chi sq sum over
    masks = get_disc_masks(LSN_template)
    
    # build a binary matrix of pixels displayed on each trial
    trial_matrix, trials_per_pixel = build_trial_matrix(LSN_template)
    
    #pre-calculate 'expected' once, since it doesn't change with shuffling 
    mean_events_per_trial = np.mean(responses,axis=0)
    expected_by_pixel = trials_per_pixel.reshape(1,num_y,num_x,2) * mean_events_per_trial.reshape(num_cells,1,1,1)#shape is (num_cells,num_y,num_x,2) 
    expected_by_pixel = expected_by_pixel.reshape(num_cells,num_y*num_x*2,1)
    
    #more convenient to keep cells as axis 0
    responses = responses.T
    
    #calculate the contribution of each pixel to the chi_sum test statistic,
    # for the actual data as well as shuffles
    chi_actual, chi_shuffle = chi_square_across_template(responses,expected_by_pixel,trial_matrix,num_shuffles)
    
    #apply region masks around each pixel to sum pixel-wise chi contributions
    # to the chi_sum test statistic, then calculate p_values
    p_values = chi_square_across_masks(masks,chi_actual,chi_shuffle,num_y,num_x)
    
    # p_values should be no smaller than the 1.0 over number of shuffles
    p_values = np.where(p_values==0.0,1.0/num_shuffles,p_values)
    
    return p_values

def get_disc_masks(LSN_template,radius=2):
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]
    LSN_binary = np.where(LSN_template==0,1,LSN_template)
    LSN_binary = np.where(LSN_binary==255,1,LSN_binary)
    LSN_binary = np.where(LSN_binary==1,1.0,0.0)
    on_trials = np.sum(LSN_binary,axis=0).astype(float)
    
    masks = np.zeros((num_y,num_x,num_y,num_x,2))
    for y in range(num_y):
        for x in range(num_x):
            trials_not_gray = np.argwhere(LSN_binary[:,y,x]>0)[:,0]
            raw_mask = np.divide(np.sum(LSN_binary[trials_not_gray,:,:],axis=0),on_trials)
            
            center_y, center_x = np.unravel_index(raw_mask.argmax(),(num_y,num_x))
            raw_mask[center_y,center_x] = 0.0
            
            x_max = center_x+radius+1
            if x_max>num_x:
                x_max=num_x
            x_min = center_x-radius
            if x_min<0:
                x_min=0
            y_max = center_y+radius+1
            if y_max>num_y:
                y_max=num_y
            y_min = center_y-radius
            if y_min<0:
                y_min=0
            
            clean_mask = np.ones(np.shape(raw_mask))
            clean_mask[y_min:y_max,x_min:x_max] = raw_mask[y_min:y_max,x_min:x_max]            

            masks[y,x,:,:,:] = clean_mask.reshape(num_y,num_x,1)
            
    masks = np.where(masks>0,0.0,1.0)
          
    masks = masks.reshape(num_y*num_x,num_y*num_x*2)
    
    return masks
      
def build_trial_matrix(LSN_template):
    num_trials = np.shape(LSN_template)[0]
    num_y = np.shape(LSN_template)[1]
    num_x = np.shape(LSN_template)[2]

    on_off_luminance = [255,0]

    trial_mat = np.zeros((num_y,num_x,2,num_trials),dtype=bool)
    for y in range(num_y):
        for x in range(num_x):
            for on_off in range(2):
                frame = np.argwhere(LSN_template[:num_trials,y,x]==on_off_luminance[on_off])[:,0]
                trial_mat[y,x,on_off,frame] = True
        

    trials_per_pixel = np.sum(trial_mat,axis=3)
    trial_mat = trial_mat.reshape(num_y*num_x*2,num_trials)
    
    return trial_mat, trials_per_pixel
    
def chi_square_across_masks(masks,chi_actual,chi_shuffle,num_y,num_x):
    
    chi_sum_actual = sum_chi(chi_actual,masks,num_y,num_x)
    chi_sum_shuffle = sum_chi(chi_shuffle,masks,num_y,num_x)
    
    p_values = (chi_sum_shuffle > chi_sum_actual).mean(axis=3)
    
    return p_values

def sum_chi(chi,masks,num_y,num_x):
    
    num_cells = np.shape(chi)[0]
    num_reps = np.shape(chi)[2]

    mask_center_pixel, pixels_in_mask = np.where(masks)
    
    chi_sum = np.zeros((num_cells,num_y*num_x,num_reps))
    for p in np.unique(mask_center_pixel):
        pixels_in_this_mask = pixels_in_mask[mask_center_pixel==p]
        chi_sum[:,p,:] = np.sum(chi[:,pixels_in_this_mask,:],axis=1)  
    
    chi_sum = chi_sum.reshape(num_cells,num_y,num_x,num_reps)
    
    return chi_sum
    
def chi_square_across_template(responses,expected_by_pixel,trial_matrix,num_shuffles):
    
    num_pixels = np.shape(trial_matrix)[0]
    num_trials = np.shape(trial_matrix)[1]
    
    (pixels,trials) = np.where(trial_matrix)
    
    # calculate pixelwise contributions to chi_sum for the actual data
    observed_by_pixel_actual = calc_observed(responses,trials.reshape(1,len(trials)),pixels,num_pixels)
    chi_actual = compute_chi(observed_by_pixel_actual,expected_by_pixel)
    
    #shuffle the trial labels to build null distribution
    trial_labels = np.random.choice(num_trials,size=(num_shuffles,num_trials))
    shuffled_trials = np.zeros((num_shuffles,len(trials))).astype(int)
    for s in range(num_shuffles):
        shuffled_trials[s,:] = trial_labels[s,trials]

    # calculate pixelwise contributions to chi_sum for the shuffled data
    observed_by_pixel_shuffled = calc_observed(responses,shuffled_trials,pixels,num_pixels)
    chi_shuffle = compute_chi(observed_by_pixel_shuffled,expected_by_pixel)
    
    return chi_actual, chi_shuffle
  
def calc_observed(responses,trials,pixels,num_pixels):
    
    num_cells = np.shape(responses)[0]
    num_reps = np.shape(trials)[0]
    
    observed_by_pixel = np.zeros((num_cells,num_pixels,num_reps))
    pixels_to_populate = np.unique(pixels)
    for p in pixels_to_populate:
        trials_pixel_is_displayed = trials[:,pixels==p]
        response_mat = responses[:,trials_pixel_is_displayed]
        observed_by_pixel[:,p,:] = response_mat.sum(axis=2)
        
    return observed_by_pixel
    
def compute_chi(observed_by_pixel,expected_by_pixel):
    
    residual_by_pixel = observed_by_pixel - expected_by_pixel  
    chi = (residual_by_pixel **2) / expected_by_pixel
    chi = np.where(expected_by_pixel>0,chi,0.0)
    
    return chi
import numpy as np
import pandas as pd
import sys
from v1dd_public import ARTIFACT_DIR
np.set_printoptions(threshold=sys.maxsize)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import joblib


csv_tag = "_events_231120"

slc2 = pd.read_csv(ARTIFACT_DIR /f'slc2_all_stim_metric{csv_tag}.csv')
slc2.insert(1, 'mouse_id', "slc2", True)

slc4 = pd.read_csv(ARTIFACT_DIR/f'slc4_all_stim_metric{csv_tag}.csv')
slc4.insert(1, 'mouse_id', "slc4", True)

slc5 = pd.read_csv(ARTIFACT_DIR/f'slc5_all_stim_metric{csv_tag}.csv')
slc5.insert(1, 'mouse_id', "slc5", True)

teto1 = pd.read_csv(ARTIFACT_DIR/f'teto1_all_stim_metric{csv_tag}.csv')
teto1.insert(1, 'mouse_id', "teto1", True)

# append 4 mice info into one pandas dataframe
cell_info = slc2.append(slc4).append(slc5).append(teto1)

cell_info["plane"].unique()
cell_info["volume"].unique()
cell_info["column"].unique()

res2 = ['frac_res_to_on',
        'frac_res_to_off',
        'frac_resp_full',
        'frac_resp_windowed',
        'frac_resp_natural_images',
        'frac_resp_natural_images_12',
        'frac_resp_natural_movie'
       ]
features = cell_info[res2]

features.rename(columns = {'frac_res_to_on': "LSN-ON",
       'frac_res_to_off': "LSN-OFF",
       'frac_resp_full':'DGF',
       'frac_resp_windowed':'DGW',
       'frac_resp_natural_images':'NI',
       'frac_resp_natural_images_12':'NI12',
       'frac_resp_natural_movie':'NM'},
        inplace = True)


all_features = [features[f] for f in features.columns.tolist()]

features_stacked = np.vstack(all_features).T

X = features_stacked

class CustomGMM(GaussianMixture): 
    def __init__(self, n_components=1, *, random_state=None, **kwargs): 
        super().__init__(n_components=n_components, random_state=random_state, **kwargs) 
        self.all_results_ = []
    def fit(self, X, y=None):
        super().fit(X, y) 
        result = {
                "n_components": self.n_components,
                "random_state": self.random_state,
                "log_likelihood": self.lower_bound_,
                "weights": self.weights_,
                "means": self.means_,
                "covariances": self.covariances_ 
                } 
        self.all_results_.append(result) 
        return self
    
    
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

min_n_components = 1
max_n_components = 50
n_inits = 100
param_grid = { 'n_components': list(range(min_n_components, max_n_components)), # Trying from 1 to 10 clusters 
              'random_state': list(range(n_inits)) # Using random_state to simulate different initializations 
              }
custom_gmm = CustomGMM() 
model = GridSearchCV(custom_gmm, param_grid, cv=4, n_jobs=-1, verbose=4, scoring=gmm_bic_score)
model.fit(X) 
# Best parameters: 
print(f"Best Number of Clusters: {model.best_params_['n_components']} with Score: {-model.best_score_:.4f}")

# joblib.dump(model, 'grid_search_1_50_clusters_100_initializations.pkl')
# joblib.dump(model, 'grid_search_40_60_clusters_100_initializations.pkl')
joblib.dump(model, 'grid_search_1_50_clusters_100_initializations_231120.pkl')
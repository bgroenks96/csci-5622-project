import numpy as np
import pandas as pd
from typing import Dict
from typing import Tuple
from problem import get_train_data
from sklearn.metrics import pairwise_distances

def stormid_dict(X_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Partitions the storm forecast dataset into separate groups for each storm and
    returns the result as a dictionary.
    """
    groups = X_df.groupby(['stormid'])
    storm_dict = dict()
    for stormid, df in groups:
        storm_dict[stormid] = df
    return storm_dict

def feature_groups(X_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions X_df into three groups by columns:
    1) 0-D features
    2) 11x11 z, u, v wind reanalysis data
    3) 11x11 sst, slp, humidity, and vorticity reanalysis data
    """
    feat_cols = X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude','hemisphere','Jday_predictor','initial_max_wind','max_wind_change_12h','dist2land'])
    nature_cols = pd.get_dummies(X_df.nature, prefix='nature', drop_first=False)
    basin_cols = pd.get_dummies(X_df.basin, prefix='basin', drop_first=False)
    X_0D = pd.concat([feat_cols, nature_cols, basin_cols], axis=1, sort=False)
    X_zuv = X_df.get([col for col in X_df.columns if col.startswith('z_') or col.startswith('u_') or col.startswith('v_')])
    X_sshv = X_df.get([col for col in X_df.columns if col.startswith('sst') or col.startswith('slp')
                   or col.startswith('hum') or col.startswith('vo700')])
    return X_0D, X_zuv, X_sshv

def trust_cont_score(X, X_map, k=10, alpha='auto'):
    """
    Computes the "trustworthiness" and "continuity" [1] of X_map with respect to X.
    This is a port and extension of the implementation provided by Van der Maaten [2].
    
    Parameters:
    X     : the data in its original representation
    X_map : the lower dimensional representation of the data to be evaluated
    k     : parameter that determines the size of the neighborhood for the T&C measure
    alpha : mixing parameter in [0,1] that determines the weight given to trustworthiness vs. continuity; higher values will give more
            weight to trustworthiness, lower values to continuity; 'auto' will compute an average across several values of alpha spanning [0,1]
    
    [1] Kaski S, Nikkilä J, Oja M, Venna J, Törönen P, Castrén E. Trustworthiness and metrics in visualizing similarity of gene expression. BMC bioinformatics. 2003 Dec;4(1):48.
    [2] Maaten L. Learning a parametric embedding by preserving local structure. InArtificial Intelligence and Statistics 2009 Apr 15 (pp. 384-391).
    """
    # Compute pairwise distance matrices
    D_h = pairwise_distances(X, X, metric='sqeuclidean')
    D_l = pairwise_distances(X_map, X_map, metric='sqeuclidean')
    # Compute neighborhood indices
    ind_h = np.argsort(D_h, axis=1)
    ind_l = np.argsort(D_l, axis=1)
    # Compute trustworthiness
    N = X.shape[0]
    T = 0
    C = 0
    t_ranks = np.zeros((k, 1))
    c_ranks = np.zeros((k, 1))
    for i in range(N):
        for j in range(k):
            t_ranks[j] = np.where(ind_h[i,:] == ind_l[i, j+1])
            c_ranks[j] = np.where(ind_l[i,:] == ind_h[i, j+1])
        t_ranks -= k
        c_ranks -= k
        T += np.sum(t_ranks[np.where(t_ranks > 0)])
        C += np.sum(c_ranks[np.where(c_ranks > 0)])
    S = (2 / (N * k * (2 * N - 3 * k - 1)))
    T = 1 - S*T
    C = 1 - S*C
    if alpha == 'auto':
        return np.average([a*T + (1-a)*C for a in np.arange(0.1, 1.0, 0.1)])
    else:
        return alpha*T + (1-alpha)*C

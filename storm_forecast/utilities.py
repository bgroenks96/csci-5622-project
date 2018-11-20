import numpy as np
import pandas as pd
from typing import Dict
from typing import Tuple

from problem import get_train_data

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
    nature_cols = pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)
    basin_cols = pd.get_dummies(X_df.basin, prefix='basin', drop_first=True)
    X_0D = pd.concat([feat_cols, nature_cols, basin_cols])
    X_zuv = X_df.get([[col for col in X_df.columns if col.startswith('z_') or col.startswith('u_') or col.startswith('v_')]])
    X_sshv = X_df.get([[col for col in X_df.columns if col.startswith('sst') or col.startswith('slp')
                   or col.startswith('hum') or col.startswith('vo700')]])
    return X_0D, X_zuv, X_sshv

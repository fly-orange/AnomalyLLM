import os
import sys
import csv
import ast
import logging
import pickle
import numpy as np
import pandas as pd
from ts_datasets.ts_datasets.anomaly.base import TSADBaseDataset
from ts_datasets.ts_datasets.anomaly.smd import download, combine_train_test_datasets

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class GECCO(TSADBaseDataset):
    """
    Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL) rover Datasets.
    SMAP and MSL are two realworld public datasets, which are two real-world datasets expert-labeled by NASA.

    - source: https://github.com/khundman/telemanom
    """

    url = "https://www.dropbox.com/s/uv9ojw353qwzqht/SMAP.tar.gz?dl=1"

    def __init__(self, subset=None, rootdir=None):
        super().__init__()

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "gecco")


        # preprocess(_logger, rootdir, dataset="SMAP")
        # Load training/test datasets
        df, metadata = load_dataset(rootdir, "GECCO")
        self.time_series.append(df)
        self.metadata.append(metadata)

def load_dataset(rootdir, dataset):
    data, label, sha = np.load(os.path.join(rootdir, f"{dataset}.npy"), allow_pickle=True)
    df = pd.DataFrame(data)
    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index*60, unit='s')
    df.index.rename("timestep", inplace=True)
    metadata = pd.DataFrame(
        {
            "trainval": df.index < df.index[int(0.7*df.shape[0])],
            "anomaly": label
        },
        index = df.index
    )

    return df, metadata
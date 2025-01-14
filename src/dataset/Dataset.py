import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.CommonUtils import get_all_file_paths, get_all_possible_subdirs, remove_dirs, save_csv

CLASS_NAME = "[DataPreparation/Dataset]"


class Dataset:

    def __init__(self, cfg):
        self.train_ratio = float(cfg["train_ratio"])
        self.test_ratio = float(cfg["test_ratio"])
        self.valid_ratio = float(cfg["valid_ratio"])
        self.seed = cfg["seed"]

        assert (self.train_ratio + self.test_ratio == 1.0) and (self.valid_ratio < self.train_ratio / 2), \
            f"{CLASS_NAME}: Error: Dataset ratio not valid." \
            f"train_ratio: [{self.train_ratio}] ; test_ratio: [{self.test_ratio}] ; valid_ratio: [{self.valid_ratio}]"

        if cfg["rmv_pre_aug"]:
            _ = [remove_dirs(get_all_possible_subdirs(i, "full_path"), "_cm") for i in cfg["input_paths"].split(",")]

        # --- List of all sample points ---#
        self.X = get_all_file_paths(cfg["input_paths"], cfg["scan_ext"])
        self.Y = get_all_file_paths(cfg["input_paths"], cfg["msk_ext"])
        self.x_ext = cfg["scan_ext"]
        self.y_ext = cfg["msk_ext"]

        # --- List of train, test and validation sets ---#
        self.train_x, self.train_y = None, None
        self.valid_x, self.valid_y = None, None
        self.test_x, self.test_y = None, None

    def generate_splits(self):
        if self.train_ratio < 1:
            x, self.test_x, y, self.test_y = train_test_split(self.X, self.Y, test_size=self.test_ratio,
                                                          random_state=self.seed)

            self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(x, y, test_size=self.test_ratio,
                                                                                  random_state=self.seed)
        else:
            self.train_x = self.X
            self.train_y = self.Y

    def save_csv(self):
        save_csv("DataFiles/", "train.csv",
                 pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': None}))
        save_csv("DataFiles/", "valid.csv",
                 pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': None}))
        save_csv("DataFiles/", "test.csv",
                 pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': None}))

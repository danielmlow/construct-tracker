"""load datasets."""

import os

import pandas as pd


def load_data(dataset: str = "reddit_27", split: str = "train") -> pd.DataFrame:
    """
    Loads data from a specified dataset and split.

    :param dataset: (str) The name of the dataset to load. Defaults to 'reddit_27'.
    :param split: (str) The split of the dataset to load. Can be 'train', 'test', or 'all'. Defaults to 'train'.
    :return: (pd.DataFrame) The loaded dataset. If the dataset or split is not found, prints an error message and returns None.
    """
    script_dir = os.path.dirname(__file__)  # Directory of the script being run
    if dataset == "reddit_27":
        if split == "train":
            return pd.read_csv(
                script_dir + "/data/datasets/reddit_27_subreddits/rmhd_27subreddits_1040posts_train.csv", index_col=0
            )

        elif split == "test":
            return pd.read_csv(
                script_dir + "/data/datasets/reddit_27_subreddits/rmhd_27subreddits_260posts_test.csv", index_col=0
            )
        elif split == "all":
            train = pd.read_csv(
                script_dir + "/data/datasets/reddit_27_subreddits/rmhd_27subreddits_1040posts_train.csv", index_col=0
            )
            test = pd.read_csv(
                script_dir + "/data/datasets/reddit_27_subreddits/rmhd_27subreddits_260posts_test.csv", index_col=0
            )
            return pd.concat([train, test], axis=0, ignore_index=True).reset_index(drop=True)
        else:
            print("split not found")
            return None

    else:
        print("dataset not found")
        return None

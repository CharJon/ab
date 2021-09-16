import numpy as np
import pandas as pd


def stats(df: pd.DataFrame, sh=10):
    s_sum = df.sum(axis=0)
    s_mean = df.mean(axis=0)
    s_median = df.median(axis=0)
    s_std = df.std(axis=0)
    s_max = df.max(axis=0)
    s_sgm = np.exp(np.sum(np.log((df + sh))) / len(df.index)) - sh

    s_min = df.min(axis=1)
    s_wins = df.eq(s_min, axis='rows').sum(axis=0)
    return pd.concat([s_sum, s_mean, s_median, s_std, s_max, s_sgm, s_wins],
                     keys=["sum", "mean", "median", "std", "max", "sgm", "wins"],
                     axis=1)


def wins_timeouts_hue(df: pd.DataFrame, time_limit, hue):
    hues = df[hue].unique()
    hue_df = wins_timeouts(df[df[hue] == hues[0]].drop(hue, axis=1), time_limit)
    hue_df[hue] = hues[0]
    hue_df.reset_index(inplace=True)
    result_df = hue_df
    for h in hues[1:]:
        hue_df = wins_timeouts(df[df[hue] == h].drop(hue, axis=1), time_limit)
        hue_df[hue] = h
        hue_df.reset_index(inplace=True)
        result_df = result_df.append(hue_df)
    return result_df


def wins_timeouts(df: pd.DataFrame, time_limit):
    s_min = df.min(axis=1)
    s_wins = df.eq(s_min, axis='rows').sum(axis=0)
    s_timeouts = df.ge(time_limit, axis='rows').sum(axis=0)
    return pd.concat([s_wins, s_timeouts],
                     keys=["wins", "timeouts"], axis=1)


def get_pivot(df, index, columns, values):
    df_pivot = df.pivot(index=index, columns=columns, values=values)
    df_pivot.reset_index(inplace=True)
    df_pivot.set_index(index, inplace=True)

    num_elements = len(df_pivot.index)
    df_pivot = df_pivot.dropna()
    num_elements_after = len(df_pivot.index)

    if num_elements - num_elements_after > 0:
        print("Care! Dropped some NaNs!")

    return df_pivot


def add_prefix_but(df, prefix, exclude=None):
    """
    Same as add_prefix, but excludes certain columns.
    :param df:
    :param prefix:
    :param exclude:
    :return:
    """
    if not exclude:
        exclude = []
    return df.rename(columns={col: f"{prefix}{col}" for col in df.columns if col not in exclude})


def merge_df(df, param, index):
    """
    Merges all sub dataframes on an index. Param is used to filter for sub dataframes.
    :param df:
    :param param:
    :param index:
    :return:
    """
    all_params = df[param].drop_duplicates().to_list()

    # prefix first
    merged_df = add_prefix_but(df[df[param] == all_params[0]], f"{all_params[0]}_", index)
    for cur_param in all_params[1:]:
        next_df = add_prefix_but(df[df[param] == cur_param], f"{cur_param}_", index)
        merged_df = merged_df.merge(next_df, on=index)

    return merged_df

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as sr
from scipy.cluster import hierarchy as hc
from typing import List, Any
import random, math

# TODO: Make categorical_cols optional argument to 
# avoid ambiguity when there are no categorical cols


def normalize_numeric(
        df, 
        cols: List[str] = []):

    tmp_df = df.copy()

    if not len(cols):
        cols = df.select_dtypes(include=[np.number]).columns

    for k in cols:
        tmp_df[k] = tmp_df[k].astype(np.float32)
        tmp_df[k] -= tmp_df[k].mean()
        tmp_df[k] /= tmp_df[k].std()

    return tmp_df

def convert_categories(
        df,
        cols: List[str] = []):

    tmp_df = df.copy()

    if not len(cols):
        cols = df.select_dtypes(include=[np.object]).columns

    tmp_df[cols] = tmp_df[cols].astype('category')
    tmp_df[cols] = tmp_df[cols].apply(lambda x: x.cat.codes)
    tmp_df[cols] = tmp_df[cols].astype('int8')

    return tmp_df

def group_by_columns(
        df: pd.DataFrame,
        columns: List[str], 
        bins: int = 6,
        categorical_cols: List[str] = [],
        ) -> pd.core.groupby.groupby.DataFrameGroupBy:

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object]).columns
    
    group_list = []
    for c in columns:
        col = df[c]
        if c in categorical_cols or not bins:
            grp = c
        else:
            col_min = col.min()
            col_max = col.max()
            # TODO: Use the original bins for display purposes as they may come normalised
            col_bins = pd.cut(col, list(np.linspace(col_min, col_max, bins)))
            grp = col_bins

        group_list.append(grp)

    grouped = df.groupby(group_list)
    return grouped 


def show_imbalance( 
        df: pd.DataFrame, 
        column_name: str, 
        cross: List[str] = [],
        categorical_cols: List[str] = [],
        bins: int = 6, 
        threshold: float = 0.5
        ) -> Any:

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object]).columns
    
    cols = cross + [column_name]
    grouped = group_by_columns(
            df,
            cols, 
            bins=bins,
            categorical_cols=categorical_cols)
    grouped_col = grouped[column_name]
    count_grp = grouped_col.count()
    count_max = count_grp.values.max()
    ratios = round(count_grp/count_max,4)
    # TODO: Make threshold a minimum number of examples per class
    imbalances = ratios < threshold
    
    cm = plt.cm.get_cmap('RdYlBu_r')
    colors = [cm(1-r/threshold/2) if t else cm(0) \
               for r,t in zip(ratios, imbalances)]
    ax = count_grp.plot.bar(color=colors)
    lp = plt.axhline(threshold*count_max, color='r')
    lp.set_label(f"Threshold: {threshold*count_max:.2f} ({threshold*100:.2f}%)")
    plt.legend()
    plt.show()
        
    return count_grp, ratios, imbalances

def show_imbalances(
        df: pd.DataFrame,
        columns: List[str] = [],
        cross: List[str] = [],
        categorical_cols: List[str] = [],
        bins: int = 6) -> Any:

    if not len(columns):
        columns = df.columns

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object]).columns

    if cross and any([x in columns for x in cross]):
        raise("Error: Columns in 'cross' are also in 'columns'")

    imbalances = []
    for col in columns:
        imbalance = show_imbalance(
            df,
            col,
            bins=bins,
            cross=cross,
            categorical_cols=categorical_cols)
        imbalances.append(imbalance)

    return imbalances

def balance(
        df: pd.DataFrame,
        column_name: str,
        cross: List[str] = [],
        upsample: int = 0.5,
        downsample: int = 1,
        bins: int = 6,
        categorical_cols: List[str] = [],
        plot: bool = True):

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object]).columns

    cols = cross + [column_name]
    grouped = group_by_columns(
                df,
                cols, 
                bins=bins,
                categorical_cols=categorical_cols)

    count_grp = grouped.count()
    count_max = count_grp.values.max()
    count_upsample = int(upsample*count_max)
    count_downsample = int(downsample*count_max)

    def norm(x):
        if x.shape[0] < count_upsample:
            return x.sample(count_upsample, replace=True)
        elif x.shape[0] > count_downsample:
            return x.sample(count_downsample)
        else:
            return x

    tmp_df = grouped.apply(norm) \
                .reset_index(drop=True)

    if plot:
        imbalance = show_imbalance(
            tmp_df,
            column_name,
            bins=bins,
            cross=cross,
            categorical_cols=categorical_cols)

    return tmp_df
    
def plot_dendogram(
        corr: pd.DataFrame, 
        cols: List[str],
        figsize=(10,5)):
    corr = np.round(corr, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method="average")
    fig = plt.figure(figsize=figsize)
    dendrogram = hc.dendrogram(
        z, labels=cols, orientation="left", leaf_font_size=16)
    plt.show()

def plot_matrix(
        corr, 
        cols: List[str], 
        figsize=(10,5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(
            corr, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(cols),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)
    plt.show()

def correlations(
        df: pd.DataFrame,
        include_categorical: bool = False,
        plot_type: str = "dendogram",
        figsize = [10,5],
        categorical_cols: List[str] = []):
    corr = None
    cols: List = []
    if include_categorical:
        corr = sr(df).correlation 
        cols = df.columns
    else:

        if not len(categorical_cols):
            categorical_cols = df.select_dtypes(include=[np.object]).columns

        cols = [c for c in df.columns if c not in categorical_cols]

        corr = df[cols].corr()
        cols = corr.columns

    if plot_type == "dendogram":
        plot_dendogram(corr, cols, figsize=figsize)
    elif plot_type == "matrix":
        plot_matrix(corr, cols, figsize=figsize)
    else:
        raise(f"Variable plot_type not valid. Provided: {plot_type}")

    return corr


def balanced_train_test_split(
        x: pd.DataFrame,
        y: np.array,
        cross: List[str] =[],
        categorical_cols: List[str] = [],
        min_per_class: int =20,
        fallback_type: str ="half",
        bins: int =6, 
        random_state: int=None,
        include_target=True):
    """
    sample_type: Can be "error", "half", or "upsample"
    """
    
    if random_state:
        random.setstate(random_state)
    
    tmp_df = x.copy()
    tmp_df["target"] = y

    # Adding target to the columns to combine
    if include_target:
        cross = ["target"] + cross
    
    if not len(categorical_cols):
        categorical_cols = list(tmp_df.select_dtypes(include=[np.object]).columns)

    # TODO: Enable for non-categorical targets
    categorical_cols = ["target"] + categorical_cols

    grouped = group_by_columns(
            tmp_df, 
            cross, 
            bins=bins,
            categorical_cols=categorical_cols)

    def resample(x):
        group_size = x.shape[0]
        if fallback_type == "upsample":
            return x.sample(min_per_class, replace=True)
        elif fallback_type == "half":
            if group_size > 2*min_per_class:
                return x.sample(min_per_class)
            else:
                if group_size > 1:
                    return x.sample(math.floor(group_size / 2))
                else:
                    if random.random() > 0.5:
                        return x
                    else:
                        return
        elif fallback_type == "error":
            if group_size > 2*min_per_class: 
                return x.sample(min_per_class)
            else:
                raise("Number of samples for group are not enough,"
                        " and fallback_type provided was 'error'")
        else:
            raise(f"Sampling type provided not found: given {fallback_type}, "\
                 "expected: 'error', 'half' or 'upsample'")
                    
    group = grouped.apply(resample)

    selected_idx = [g[-1] for g in group.index.values]
    
    train_idx = np.full(tmp_df.shape[0], True, dtype=bool)
    train_idx[selected_idx] = False
    test_idx = np.full(tmp_df.shape[0], False, dtype=bool)
    test_idx[selected_idx] = True
    
    df_train = tmp_df.iloc[train_idx] 
    df_test = tmp_df.iloc[test_idx]

    x_train = df_train.drop("target", axis=1)
    y_train = df_train["target"].values
    x_test = df_test.drop("target", axis=1)
    y_test = df_test["target"].values
    
    return x_train, y_train, x_test, y_test



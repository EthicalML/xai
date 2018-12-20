import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as sr
from scipy.cluster import hierarchy as hc
from typing import List, Any
import random, math
# TODO: Remove Dependencies, starting with Sklearn
from sklearn.metrics import roc_curve, precision_recall_curve

# TODO: Make categorical_cols optional argument (None) to 
# avoid ambiguity when there are no categorical cols


def normalize_numeric(
        df, 
        numerical_cols: List[str] = []):

    tmp_df = df.copy()

    if not len(numerical_cols):
        numerical_cols = df.select_dtypes(include=[np.number]).columns

    for k in numerical_cols:
        tmp_df[k] = tmp_df[k].astype(np.float32)
        tmp_df[k] -= tmp_df[k].mean()
        tmp_df[k] /= tmp_df[k].std()

    return tmp_df

def convert_categories(
        df,
        categorical_cols: List[str] = []):

    tmp_df = df.copy()

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object]).columns

    tmp_df[categorical_cols] = tmp_df[categorical_cols].astype('category')
    tmp_df[categorical_cols] = tmp_df[categorical_cols].apply(lambda x: x.cat.codes)
    tmp_df[categorical_cols] = tmp_df[categorical_cols].astype('int8')

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
    sample_type: Can be "error", or "half""
    """
    # TODO: Allow parameter test_size:int so it's possible 
    # to provide preferred test size, and fill up the rest with normal .sample()
    
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
        if fallback_type == "half":
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
                 "expected: 'error', or 'half'")
                    
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



def convert_probs(probs, threshold=0.5):
    """Convert probabilities into classes"""
    # TODO: Enable for multiclass
    return (probs >= threshold).astype(int)

def perf_metrics(y_valid, y_pred):
    TP = np.sum( y_pred[y_valid==1] )
    TN = np.sum( y_pred[y_valid==0] == 0 )
    FP = np.sum(  y_pred[y_valid==0] )
    FN = np.sum(  y_pred[y_valid==1] == 0 )

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
    accuracy = (TP+TN) / (TP+TN+FP+FN)

    return precision, recall, specificity, accuracy

def metrics_imbalance(
        x_df,
        y_valid,
        y_pred,
        col_name=None,
        cross=[],
        categorical_cols=[],
        bins=6,
        prob_threshold=0.5,
        plot=True):

    x_tmp = x_df.copy()
    x_tmp["target"] = y_valid
    x_tmp["predicted"] = y_pred

    # Convert predictions into classes
    # TODO: Enable for multiclass
    if x_tmp["predicted"].dtype.kind == 'f':
        x_tmp["predicted"] = convert_probs(
            x_tmp["predicted"], threshold=prob_threshold)

    if col_name is None:
        grouped = [("target", x_tmp),]
    else:
        cols = cross + [col_name]
        grouped = group_by_columns(
            x_tmp,
            cols,
            bins=bins,
            categorical_cols=categorical_cols)

    prfs = []
    classes = []
    for group, group_df in grouped:
        group_valid = group_df["target"].values
        group_pred = group_df["predicted"].values
        precision, recall, specificity, accuracy = \
            perf_metrics(group_valid, group_pred)
        prfs.append([precision, recall, specificity, accuracy])
        classes.append(str(group))

    prfs_cols = ["precision", "recall", "specificity", "accuracy"]
    prfs_df = pd.DataFrame(
            prfs, 
            columns=prfs_cols, 
            index=classes)

    if plot:
        prfs_df.plot.bar(figsize=(20,5))
        lp = plt.axhline(0.5, color='r')
        lp = plt.axhline(1, color='g')

    return prfs_df

def metrics_imbalances(
        x_test,
        y_test,
        predictions,
        columns=[],
        categorical_cols=[],
        cross=[],
        bins=6,
        prob_threshold=0.5,
        plot=True):

    if not len(columns):
        columns = x_test.columns

    if not len(categorical_cols):
        categorical_cols = x_test.select_dtypes(include=[np.object]).columns

    results = []
    for col in columns:
        r = metrics_imbalance(
            x_test,
            y_test,
            predictions,
            col,
            cross=cross,
            categorical_cols=categorical_cols,
            bins=6,
            prob_threshold=prob_threshold,
            plot=True)
        results.append(r)

    return results



def roc_imbalance(
        x_df,
        y_valid,
        y_pred,
        col_name=None,
        cross=[],
        categorical_cols=None,
        bins=6,
        plot=True):

    x_tmp = x_df.copy()
    x_tmp["target"] = y_valid
    x_tmp["predicted"] = y_pred

    if col_name is None:
        grouped = [("target", x_tmp),]
    else:
        cols = cross + [col_name]
        grouped = group_by_columns(
            x_tmp,
            cols,
            bins=bins,
            categorical_cols=categorical_cols)

    if plot:
        plt.figure()

    fprs = tprs = []

    for group, group_df in grouped:
        group_valid = group_df["target"]
        group_pred = group_df["predicted"]

        fpr, tpr, _ = roc_curve(group_valid, group_pred)
        fprs.append(fpr)
        tprs.append(tpr)

        if plot:
            plt.plot(fpr, tpr, label=group)
            plt.plot([0, 1], [0, 1], 'k--')

    if plot:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    return fprs, tprs

def roc_imbalances(
        x_test,
        y_test,
        predictions,
        columns=[],
        categorical_cols=[],
        cross=[],
        bins=6,
        plot=True):

    if not len(columns):
        columns = x_test.columns

    if not len(categorical_cols):
        categorical_cols = x_test.select_dtypes(include=[np.object]).columns

    results = []
    for col in columns:
        r = roc_imbalance(
            x_test,
            y_test,
            predictions,
            col,
            cross=cross,
            categorical_cols=categorical_cols,
            bins=6,
            plot=True)
        results.append(r)

    return results



def pr_imbalance(
        x_df,
        y_valid,
        y_pred,
        col_name=None,
        cross=[],
        categorical_cols=None,
        bins=6,
        plot=True):

    x_tmp = x_df.copy()
    x_tmp["target"] = y_valid
    x_tmp["predicted"] = y_pred

    if col_name is None:
        grouped = [("target", x_tmp),]
    else:
        cols = cross + [col_name]
        grouped = group_by_columns(
            x_tmp,
            cols,
            bins=bins,
            categorical_cols=categorical_cols)

    if plot:
        plt.figure()

    prs = rcs = []

    for group, group_df in grouped:
        group_valid = group_df["target"]
        group_pred = group_df["predicted"]

        pr, rc, _ = precision_recall_curve(group_valid, group_pred)
        prs.append(pr)
        rcs.append(rc)

        if plot:
            plt.plot(pr,rc, label=group)

    if plot:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.show()

    return prs, rcs

def pr_imbalances(
        x_test,
        y_test,
        predictions,
        columns=[],
        categorical_cols=[],
        cross=[],
        bins=6,
        plot=True):

    if not len(columns):
        columns = x_test.columns

    if not len(categorical_cols):
        categorical_cols = x_test.select_dtypes(include=[np.object]).columns

    results = []
    for col in columns:
        r = pr_imbalance(
            x_test,
            y_test,
            predictions,
            col,
            cross=cross,
            categorical_cols=categorical_cols,
            bins=6,
            plot=True)
        results.append(r)

    return results


def smile_imbalance(
        y_test, 
        probs, 
        threshold=0.5, 
        manual_review=None,
        display_breakdown=False,
        bins=10):
    
    # TODO: Change function so it only iterates once

    preds = convert_probs(probs, threshold).flatten()
    d = pd.DataFrame(probs)
    d.columns = ["probs"]
    d["preds"] = preds
    d["target"] = y_test

    tps = np.full(y_test.shape, False, bool)

    d["true-positives"] = np.full(y_test.shape[0], False, bool)
    d["true-negatives"] = np.full(y_test.shape[0], False, bool)
    d["false-positives"] = np.full(y_test.shape[0], False, bool)
    d["false-negatives"] = np.full(y_test.shape[0], False, bool)
    d["manual-review"] =  np.full(y_test.shape[0], False, bool)

    d["true-positives"].loc[y_test == 1] = preds[y_test == 1] == 1
    d["true-negatives"].loc[y_test == 0] = preds[y_test == 0] == 0
    d["false-positives"].loc[y_test == 0] = preds[y_test == 0] == 1
    d["false-negatives"].loc[y_test == 1] = preds[y_test == 1] == 0

    d["correct"] = d["true-positives"].values
    d["correct"].loc[d["true-negatives"] == 1] = True

    d["incorrect"] = d["false-positives"].values
    d["incorrect"].loc[d["false-negatives"] == 1] = True 
    
    if display_breakdown:
        disp_cols = ["true-positives", 
                     "true-negatives", 
                     "false-positives", 
                     "false-negatives"]
    else:
        disp_cols = ["correct", "incorrect"]
    
    if manual_review:
        gt = probs > manual_review
        lt = probs < threshold
        d["manual-review"] = gt * lt > 0
        
        if display_breakdown:
            d["true-positives"].loc[d["manual-review"]] = False
            d["true-negatives"].loc[d["manual-review"]] = False
            d["false-positives"].loc[d["manual-review"]] = False
            d["false-negatives"].loc[d["manual-review"]] = False
        else:
            d["correct"].loc[d["manual-review"]] = False
            d["incorrect"].loc[d["manual-review"]] = False
        
        disp_cols.append("manual-review")

    d["true-positives"] = d["true-positives"].astype(int) 
    d["true-negatives"] = d["true-negatives"].astype(int)
    d["false-positives"] = d["false-positives"].astype(int)
    d["false-negatives"] = d["false-negatives"].astype(int)
    d["correct"] = d["correct"].astype(int)
    d["incorrect"] = d["incorrect"].astype(int)

    grouped = group_by_columns(d, ["probs"], bins=bins)

    ax = grouped[disp_cols].sum().plot.bar(stacked=True, figsize=(15,5))
    lim = ax.get_xlim()
    ran = lim[1] - lim[0]
    thre = ran*threshold + lim[0]
    plt.axvline(thre)
    if manual_review:
        manr = ran*manual_review + lim[0]
        plt.axvline(manr)
    # TODO: Need to fix this hack and use the index
    ax_xticks = [label.get_text().split()[1][:-1] for label in ax.get_xticklabels()]
    ax.set_xticklabels(ax_xticks)
    
    return d


def feature_importance(x, y, func, repeat=10, plot=True):
    base_score = func(x, y)
    imp = [0] * len(x.columns)
    for i in range(repeat):
        for j, c in enumerate(x.columns):
            tmp = x[c].values.copy()
            np.random.shuffle(x[c].values)
            score = func(x, y)
            x[c] = tmp
            imp[j] += base_score - score
    imp = [a/repeat for a in imp]
    imp_df = pd.DataFrame(data=[imp], columns=x.columns)
    if plot:
        imp_df.sum().sort_values().plot.barh()
    return imp_df






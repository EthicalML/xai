import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as sr
from scipy.cluster import hierarchy as hc
from typing import List, Any, Union, Tuple
import random, math
# TODO: Remove Dependencies, starting with Sklearn
from sklearn.metrics import roc_curve, \
        precision_recall_curve, roc_auc_score, \
        confusion_matrix
import itertools
import logging

# TODO: Make categorical_cols optional argument (None) to 
# avoid ambiguity when there are no categorical cols


def normalize_numeric(
        df, 
        numerical_cols: List[str] = []):
    """
    Normalizes numeric columns by substracting the mean and dividing
        by standard deviation. If the parameter numerical_cols is not
        provided, it will take all the columns of dtype np.number.

    :Example:

    norm_df = xai.normalize_numeric(
                df,
                normalize_numeric=["age", "other_numeric_attribute"])

    :param df: Pandas Dataframe containing data (inputs and target)
    :type df: pd.DataFrame
    :param numerical_cols: List of strings containing numercial cols
    :type categorical_cols: str
    :returns: Dataframe with normalized numerical values.
    :rtype: pandas.DataFrame

    """
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
    """
    Converts columns to numeric categories. If the categorical_cols
        parameter is passed as a list then those columns are converted.
        Otherwise, all np.object columns are converted.

    :Example:

    import xai
    cat_df = xai.convert_categories(df)

    :param df: Pandas Dataframe containing data (inputs and target)
    :type df: pandas.DataFrame
    :param categorical_cols: List of strings containing categorical cols
    :type categorical_cols: str
    :returns: Dataframe with categorical numerical values.
    :rtype: pandas.DataFrame

    """
    tmp_df = df.copy()

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object, np.bool]).columns

    tmp_df[categorical_cols] = tmp_df[categorical_cols].astype('category')
    tmp_df[categorical_cols] = tmp_df[categorical_cols].apply(lambda x: x.cat.codes)
    tmp_df[categorical_cols] = tmp_df[categorical_cols].astype('int8')

    return tmp_df

def group_by_columns(
        df: pd.DataFrame,
        columns: List[str], 
        bins: int = 6,
        categorical_cols: List[str] = []):
    """
    Groups dataframe by the categories (or bucketized values) for all columns provided. 
        If categorical it uses categories,
        if numeric, it uses bins. If more than one column is provided, the columns
        provided are, for example, age and binary_target_label, then the result 
        would be a pandas DataFrame that is grouped by age groups for each of the
        positive and negative/positive labels.

    :Example:

    columns=["loan", "gender"]
    df_groups = xai.group_by_columns(
        df, 
        columns=columns,
        bins=10,
        categorical_cols=["gender", "loan"])

    for group, df_group in df_groups:
        print(group) 
        print(grouped_df.head())

    :param df: Pandas Dataframe containing data (inputs and target)
    :type df: pandas.DataFrame
    :param bins: [Default: 6] Number of bins to be used for numerical cols
    :type bins: int
    :param categorical_cols: [Default: []] Columns within dataframe that are
        categorical. Columns that are not np.objects or np.bool and 
        are not part explicitly
        provided here will be treated as numeric, and bins will be used.
    :type categorical_cols: List[str]
    :returns: Dataframe with categorical numerical values.
    :rtype: pandas.core.groupby.groupby.DataFrameGroupBy

    """

    if not len(categorical_cols):
        categorical_cols = _infer_categorical(df)
    
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


def imbalance_plot( 
        df: pd.DataFrame, 
        *cross_cols: str, 
        categorical_cols: List[str] = [],
        bins: int = 6, 
        threshold: float = 0.5):
    """
    Shows the number of examples provided for each of the values across the
    product tuples in the columns provided. If you would like to do processing
    with the sub-groups created by this class please see the 
    group_by_columns function.

    :Example:

    import xai
    class_counts = xai.imbalance_plot(
        df, 
        "gender", "loan",
        bins=10,
        threshold=0.8)

    :param df: Pandas Dataframe containing data (inputs and target)
    :type df: pandas.DataFrame
    :param *cross_cols: One or more positional arguments (passed as *args) that 
    are used to split the data into the cross product of their values 
    :type cross: List[str]
    :param categorical_cols: [Default: []] Columns within dataframe that are
        categorical. Columns that are not np.objects and are not part explicitly
        provided here will be treated as numeric, and bins will be used.
    :type categorical_cols: List[str]
    :param bins: [Default: 6] Number of bins to be used for numerical cols
    :type bins: int
    :param threshold: [Default: 0.5] Threshold to display in the chart.
    :type bins: float
    :returns: Null
    :rtype: None

    """

    if not cross_cols:
        raise TypeError("imbalance_plot requires at least 1 string column name")

    grouped = group_by_columns(
            df,
            list(cross_cols), 
            bins=bins,
            categorical_cols=categorical_cols)

    grouped_col = grouped[cross_cols[0]]
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

def balance(
        df: pd.DataFrame,
        *cross_cols: str, 
        upsample: float = 0.5,
        downsample: int = 1,
        bins: int = 6,
        categorical_cols: List[str] = [],
        plot: bool = True):
    """
    Balances a dataframe based on the columns and cross columns provided.
        The results can be upsampled or downsampled. By default, there is no
        downsample, and the upsample is towards a minimum of 50% of the 
        frequency of the highest class.

    :Example:

    cat_df = xai.balance(
        df, 
        "gender", "loan",
        upsample=0.8,
        downsample=0.8)

    :param df: Pandas Dataframe containing data (inputs and target )
    :type df: pandas.DataFrame
    :param *cross_cols: One or more positional arguments (passed as *args) that 
    are used to split the data into the cross product of their values 
    :type cross: List[str]
    :param upsample: [Default: 0.5] Target upsample for columns lower 
        than percentage.
    :type upsample: float
    :param downsample: [Default: 1] Target downsample for columns higher 
        than percentage.
    :type downsample: float
    :param bins: [Default: 6] Number of bins to be used for numerical cols
    :type bins: int
    :param categorical_cols: [Default: []] Columns within dataframe that are
        categorical. Columns that are not np.objects and are not part explicitly
        provided here will be treated as numeric, and bins will be used.
    :type categorical_cols: List[str]
    :param threshold: [Default: 0.5] Threshold to display in the chart.
    :type bins: float
    :returns: Dataframe with categorical numerical values.
    :rtype: pandas.DataFrame

    """

    if not len(categorical_cols):
        categorical_cols = df.select_dtypes(include=[np.object, np.bool]).columns

    grouped = group_by_columns(
                df,
                list(cross_cols), 
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
        imbalance_plot(
            tmp_df,
            *cross_cols,
            bins=bins,
            categorical_cols=categorical_cols)

    return tmp_df
    
def _plot_correlation_dendogram(
        corr: pd.DataFrame, 
        cols: List[str],
        plt_kwargs={}):
    """
    Plot dendogram of a correlation matrix, using the columns provided. 
    This consists of a chart that that shows hierarchically the variables
    that are most correlated by the connecting trees. The closer to the right
    that the connection is, the more correlated the features are.
    If you would like to visualise this as a tree, please 
    see the function _plot_correlation_dendogram.

    :Example:

    columns_to_include=["age", "loan", "gender"]
    xai._plot_correlation_dendogram(df, cols=columns_to_include)

    :returns: Null
    :rtype: None

    """

    corr = np.round(corr, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method="average")
    fig = plt.figure(**plt_kwargs)
    dendrogram = hc.dendrogram(
        z, labels=cols, orientation="left", leaf_font_size=16)
    plt.show()

def _plot_correlation_matrix(
        corr, 
        cols: List[str], 
        plt_kwargs={}):
    """
    Plot a matrix of the correlation matrix, using the columns provided in params. 
    This visualisation contains all the columns in the X and Y axis, where the 
    intersection of the column and row displays the correlation value. 
    The closer this correlation factor is to 1, the more correlated the features
    are. If you would like to visualise this as a tree, please see 
    the function _plot_correlation_dendogram.

    :Example:

    columns_to_include=["age", "loan", "gender"]
    xai._plot_correlation_matrix(df, cols=columns_to_include)

    :returns: Null
    :rtype: None

    """
    fig = plt.figure(**plt_kwargs)
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
        plt_kwargs={},
        categorical_cols: List[str] = []):

    corr = None
    cols: List = []
    if include_categorical:
        corr = sr(df).correlation 
        cols = df.columns
    else:

        if not len(categorical_cols):
            categorical_cols = df.select_dtypes(include=[np.object, np.bool]).columns

        cols = [c for c in df.columns if c not in categorical_cols]

        corr = df[cols].corr()
        cols = corr.columns

    if plot_type == "dendogram":
        _plot_correlation_dendogram(corr, cols, plt_kwargs=plt_kwargs)
    elif plot_type == "matrix":
        _plot_correlation_matrix(corr, cols, plt_kwargs=plt_kwargs)
    else:
        raise ValueError(f"Variable plot_type not valid. Provided: {plot_type}")

    return corr

def confusion_matrix_plot(
        y_test, 
        pred, 
        scaled=True,
        label_x_neg="PREDICTED NEGATIVE",
        label_x_pos="PREDICTED POSITIVE", 
        label_y_neg="ACTUAL NEGATIVE", 
        label_y_pos="ACTUAL POSITIVE"):

    confusion = confusion_matrix(y_test, pred)
    columns = [label_y_neg, label_y_pos]
    index = [label_x_neg, label_x_pos]

    if scaled:
        confusion_scaled = (confusion.astype("float") / 
                            confusion.sum(axis=1)[:, np.newaxis])
        confusion = pd.DataFrame(
                confusion_scaled, 
                index=index, 
                columns=columns)
    else:
        confusion = pd.DataFrame(
                confusion,
                index=index,
                columns=columns)

    cmap = plt.get_cmap("Blues")
    plt.figure()
    plt.imshow(confusion, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()

    plt.xticks(np.arange(2), columns, rotation=45)
    plt.yticks(np.arange(2), index, rotation=45)

    threshold = 0.5 if scaled else confusion.max().max() / 2
    for i, j in itertools.product(
            range(confusion.shape[0]), 
            range(confusion.shape[1])):
        txt = "{:,}".format(confusion.iloc[i,j])
        if scaled: txt = "{:0.4f}".format(confusion.iloc[i,j])
        plt.text(j, i, txt,
                    horizontalalignment="center",
                    color=("white" if confusion.iloc[i,j] > threshold else "black"))

    plt.tight_layout()
    plt.show()


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
        categorical_cols = list(tmp_df.select_dtypes(include=[np.object, np.bool]).columns)

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
    
    return x_train, y_train, x_test, y_test, train_idx, test_idx


def convert_probs(probs, threshold=0.5):
    """Convert probabilities into classes"""
    # TODO: Enable for multiclass
    return (probs >= threshold).astype(int)

def evaluation_metrics(y_valid, y_pred):

    TP = np.sum( y_pred[y_valid==1] )
    TN = np.sum( y_pred[y_valid==0] == 0 )
    FP = np.sum(  y_pred[y_valid==0] )
    FN = np.sum(  y_pred[y_valid==1] == 0 )

    # Adding an OR to ensure it doesn't divide by zero
    precision = TP / ((TP+FP) or 0.001)
    recall = TP / ((TP+FN) or 0.001)
    specificity = TN / ((TN+FP) or 0.001)
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    f1 = 2 * (precision * recall) / ((precision + recall) or 0.001)
    try:
        auc = roc_auc_score(y_valid, y_pred)
    except ValueError:
        auc = 0

    return {
        "precision": precision, 
        "recall": recall, 
        "specificity": specificity, 
        "accuracy": accuracy, 
        "auc": auc, 
        "f1": f1
    }

def metrics_plot(
        target,
        predicted,
        df=pd.DataFrame(),
        cross_cols=[],
        categorical_cols=[],
        bins=6,
        plot=True,
        exclude_metrics=[],
        plot_threshold=0.5):

    grouped = _group_metrics(
        target,
        predicted,
        df, 
        cross_cols, 
        categorical_cols,
        bins,
        target_threshold=plot_threshold)

    prfs = []
    classes = []
    for group, group_df in grouped:
        group_valid = group_df["target"].values
        group_pred = group_df["predicted"].values
        metrics_dict = \
            evaluation_metrics(group_valid, group_pred)
        # Remove metrics as specified by params
        [metrics_dict.pop(k, None) for k in exclude_metrics]
        prfs.append(list(metrics_dict.values()))
        classes.append(str(group))

    prfs_cols = metrics_dict.keys()
    prfs_df = pd.DataFrame(
            np.array(prfs).transpose(), 
            columns=classes, 
            index=prfs_cols)

    if plot:
        prfs_df.plot.bar(figsize=(20,5))
        lp = plt.axhline(0.5, color='r')
        lp = plt.axhline(1, color='g')

    return prfs_df

def roc_plot(
        target,
        predicted,
        df=pd.DataFrame(),
        cross_cols=[],
        categorical_cols=[],
        bins=6,
        plot=True):

    return _curve(
        target=target,
        predicted=predicted,
        curve_type="roc",
        df=df,
        cross_cols=cross_cols,
        categorical_cols=categorical_cols,
        bins=bins,
        plot=plot)

def pr_plot(
        target,
        predicted,
        df=pd.DataFrame(),
        cross_cols=[],
        categorical_cols=[],
        bins=6,
        plot=True):

    return _curve(
        target=target,
        predicted=predicted,
        curve_type="pr",
        df=df,
        cross_cols=cross_cols,
        categorical_cols=categorical_cols,
        bins=bins,
        plot=plot)

def _curve(
        target,
        predicted,
        curve_type="roc",
        df=pd.DataFrame(),
        cross_cols=[],
        categorical_cols=[],
        bins=6,
        plot=True):

    if curve_type == "roc":
        curve_func = roc_curve
        y_label = 'False Positive Rate'
        x_label = 'True Positive Rate'
        p1 = [0,1]
        p2 = [0,1]
        y_lim = [0, 1.05]
        legend_loc = "lower right"
    elif curve_type == "pr":
        curve_func = precision_recall_curve 
        y_label = "Recall"
        x_label = "Precision"
        p1 = [1,0]
        p2 = [0.5,0.5]
        y_lim = [0.25, 1.05]
        legend_loc = "lower left"
    else:
        raise ValueError("Curve function provided not valid. "
                f" curve_func provided: {curve_func}")

    grouped = _group_metrics(
        target,
        predicted,
        df, 
        cross_cols, 
        categorical_cols,
        bins)

    if plot:
        plt.figure()

    r1s = r2s = []

    for group, group_df in grouped:
        group_valid = group_df["target"]
        group_pred = group_df["predicted"]

        r1, r2, _ = curve_func(group_valid, group_pred)
        r1s.append(r1)
        r2s.append(r2)

        if plot:
            if curve_type == "pr": r1,r2 = r2,r1
            plt.plot(r1, r2, label=group)
            plt.plot(p1, p2, 'k--')

    if plot:
        plt.xlim([0.0, 1.0])
        plt.ylim(y_lim)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc=legend_loc)
        plt.show()

    return r1s, r2s


def _infer_categorical(df):
    categorical_cols = df.select_dtypes(
            include=[np.object, np.bool, np.int8]).columns
    logging.warn("No categorical_cols passed so inferred using np.object, "
            f"np.int8 and np.bool: {categorical_cols}. If you see an error"
            " these are not "
            "correct, please provide them as a string array as: "
            "categorical_cols=['col1', 'col2', ...]")
    return categorical_cols


def _group_metrics(
        target,
        predicted,
        df, 
        cross_cols, 
        categorical_cols,
        bins,
        target_threshold=None):

    if not all(c in df.columns for c in cross_cols):
        raise KeyError("Cross columns don't match columns in dataframe provided.")

    df_tmp = df.copy()
    df_tmp["target"] = target
    df_tmp["predicted"] = predicted

    # Convert predictions into classes
    if target_threshold and df_tmp["predicted"].dtype.kind == 'f':
        df_tmp["predicted"] = convert_probs(
            df_tmp["predicted"], threshold=target_threshold)

    if categorical_cols and cross_cols:
        categorical_cols = _infer_categorical(df_tmp)

    if not cross_cols:
        grouped = [("target", df_tmp),]
    else:
        grouped = group_by_columns(
            df_tmp,
            cross_cols,
            bins=bins,
            categorical_cols=categorical_cols)

    return grouped



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






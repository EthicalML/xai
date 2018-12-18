import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as sr
from scipy.cluster import hierarchy as hc
from typing import List, Any
import math

class XData:
    def __init__(self, 
                target_name: str,
                df: pd.DataFrame = None,
                csv_path: str = None,
                protected_cols: List[str] = None,
                col_names: List[str] = None,
                categorical_cols: List[str] = None,
                string_cols: List[str] = None,
                numerical_cols: List[str] = None,
                threshold: float = 0.5,
                **kwargs) -> None:
        
        self._threshold = threshold
        self._target_name = target_name

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise Exception("df parameter provided, but not of type dataframe.")
        else:
            if not csv_path:
                raise Exception("Neither df or csv_file parameters provided.")
            df = self._read_csv(csv_path, col_names, **kwargs)

        self._initialise_from_dataframe(df, col_names, categorical_cols)

        if not categorical_cols:
            self._categorical_cols = self.df.select_dtypes(include=[np.object]).columns

        if not numerical_cols:
            self._numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        if not protected_cols:
            self._protected_cols = self.df.columns

    def _initialise_from_dataframe(self,
            df: pd.DataFrame,
            col_names: List[str] = None,
            categorical_cols: List[str] = None) -> None:
        """
        Process dataframe provided to create XData object from it. 
        """
        if col_names:
            df.columns = col_names
        self.orig_df = df.copy()
        self.df = df

    def set_protected(self, 
            protected_cols: List[str]):
        if any([c not in self.df.columns for c in protected_cols]):
            raise("One (or more) of the columns provided do not exist."\
                    " Provided: {protected_cols}. Actual: {self.df.columns}")
        self._protected_cols = protected_cols

    def _read_csv(self, 
            csv_path: str, 
            col_names: List[str] = None, 
            **kwargs) -> pd.DataFrame:
        """
        Read dataframe from csv to pandas dataframe and return result.
        """

        df = pd.read_csv(csv_path, names=col_names, **kwargs)
        return df

    def set_threshold(self, threshold):
        self._threshold = threshold

    def reset(self):
        self.df = self.orig_df.copy()

    def normalize_numeric(self):
        numerical_cols = self._numerical_cols
        for k in numerical_cols:
            self.df[k] = self.df[k].astype(np.float32)
            self.df[k] -= self.df[k].mean()
            self.df[k] /= self.df[k].std()

        return self.df

    def convert_categories(self,
            categorical_cols: List[str] = None):
        if categorical_cols is None:
            categorical_cols = self._categorical_cols
        self.df[categorical_cols] = self.df[categorical_cols].astype('category')
        self.df[categorical_cols] = self.df[categorical_cols].apply(lambda x: x.cat.codes)
        self.df[categorical_cols] = self.df[categorical_cols].astype('int8')

        return self.df

    def _group_by_columns(self,
            all_cols: List[str], 
            bins: int) -> pd.core.groupby.groupby.DataFrameGroupBy:
        group_list = []
        for c in all_cols:
            col = self.df[c]
            if c in self._categorical_cols or not bins:
                grp = c
            else:
                col_min = col.min()
                col_max = col.max()
                # TODO: Use the original bins for display purposes as they may come normalised
                col_bins = pd.cut(col, list(np.linspace(col_min, col_max, bins)))
                grp = col_bins

            group_list.append(grp)

        grouped = self.df.groupby(group_list)
        return grouped 


    def show_imbalance(self, 
            column_name: str, 
            bins: int = 10, 
            cross: List[str] = None) -> Any:

        if cross is None:
            cross = [self._target_name]

        all_cols = cross + [column_name]
        grouped = self._group_by_columns(all_cols, bins)
        grouped_col = grouped[column_name]
        count_grp = grouped_col.count()
        count_max = count_grp.values.max()
        ratios = round(count_grp/count_max,4)
        imbalances = ratios < self._threshold
        
        cm = plt.cm.get_cmap('RdYlBu_r')
        colors = [cm(1-r/self._threshold/2) if t else cm(0) \
                   for r,t in zip(ratios, imbalances)]
        ax = count_grp.plot.bar(color=colors)
        lp = plt.axhline(self._threshold*count_max, color='r')
        lp.set_label(f"Threshold: {self._threshold*count_max:.2f} ({self._threshold*100:.2f}%)")
        plt.legend()
        plt.show()
            
        return count_grp, ratios, imbalances

    def show_imbalances(self,
            bins: int = 10,
            column_names: List[str] = [],
            cross: List[str] = None) -> Any:

        # TODO: Ensure column_names and cross are part of df.columns
        print(self._protected_cols)

        if not column_names:
            column_names = self._protected_cols

        if cross and any([x in column_names for x in cross]):
            raise("Error: Columns in 'cross' are also in 'column_names'")

        if cross is None:
            cross = [self._target_name]

        imbalances = []
        for col in column_names:
            imbalance = self.show_imbalance(
                col,
                bins=bins,
                cross=cross)
            imbalances.append(imbalance)

        return imbalances

    def balance(self,
            column_name: str,
            cross: List[str] = None,
            upsample: int = None,
            downsample: int = 1,
            bins: int = None):

        if cross is None:
            cross = [self._target_name]

        if not upsample:
            upsample = self._threshold

        all_cols = cross + [column_name]
        grouped = self._group_by_columns(all_cols, bins)

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

        self.df = grouped.apply(norm) \
                    .reset_index(drop=True)
        
    def _plot_dendogram(self, 
            corr, 
            cols: List[str],
            figsize=(10,5)):
        corr = np.round(corr, 4)
        corr_condensed = hc.distance.squareform(1-corr)
        z = hc.linkage(corr_condensed, method="average")
        fig = plt.figure(figsize=figsize)
        dendrogram = hc.dendrogram(
            z, labels=cols, orientation="left", leaf_font_size=16)
        plt.show()

    def _plot_matrix(self, 
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

    def correlations(self, 
            include_categorical: bool = False,
            plot_type: str = "dendogram",
            figsize = [10,5]):
        corr = None
        cols: List = []
        if include_categorical:
            corr = sr(self.df).correlation 
            cols = self.df.columns
        else:
            corr = self.df[self._numerical_cols].corr()
            cols = corr.columns

        if plot_type == "dendogram":
            self._plot_dendogram(corr, cols, figsize=figsize)
        elif plot_type == "matrix":
            self._plot_matrix(corr, cols, figsize=figsize)
        else:
            raise(f"Variable plot_type not valid. Provided: {plot_type}")

        return corr
    


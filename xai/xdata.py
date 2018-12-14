import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as sr
from scipy.cluster import hierarchy as hc
from typing import List, Any
import math

class XData:
    def __init__(self, 
                df: pd.DataFrame = None,
                csv_path: str = None,
                col_names: List[str] = None,
                categorical_cols: List[str] = None,
                string_cols: List[str] = None,
                numerical_cols: List[str] = None,
                threshold: int = 0.5,
                **kwargs) -> None:
        
        self._threshold = threshold

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise Exception("df parameter provided, but not of type dataframe.")
        else:
            if not csv_path:
                raise Exception("Neither df or csv_file parameters provided.")
            df = self._read_csv(csv_path, col_names, **kwargs)

        if not categorical_cols:
            self._categorical_cols = self.df.select_dtypes(include=[np.object])

        if not numerical_cols:
            self._numerical_cols = self.df.select_dtypes(include=[np.number])

        self._initialise_from_dataframe(df, col_names, categorical_cols)

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
        self.df = self.orig_df

    def normalize_numeric(self):
        numerical_cols = self._numerical_cols
        for k in numerical_cols:
            self.df[k] = self.df[k].astype(np.float32)
            self.df[k] -= self.df[k].mean()
            self.df[k] /= self.df[k].std()

        return self.df

    def convert_categories(self):
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
            if col.dtype == np.object or not bins:
                grp = c
            else:
                col_min = col.min()
                col_max = col.max()
                col_bins = pd.cut(col, list(range(col_min, col_max, 
                    math.ceil((col_max-col_min)/bins))))
                grp = col_bins

            group_list.append(grp)

        grouped = self.df.groupby(group_list)
        return grouped 


    def imbalance(self, 
            column_name: str, 
            bins: int = 10, 
            cross_column_names: List[str] = []) -> None:

        all_cols = cross_column_names + [column_name]
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
            
        return None

    def imbalances(self,
            bins: int = 10,
            column_names: List[str] = [],
            cross_column_names: List[str] = []) -> None:

        # TODO: Ensure column_names and cross_column_names are part of df.columns
        if not column_names:
            column_names = [x for x in list(self.df.columns) if x not in cross_column_names]
        else:
            if any([x in column_names for x in cross_column_names]):
                raise("Error: Columns in 'cross_column_names' are also in 'column_names'")

        imbalances = []
        for col in column_names:
            imbalance = self.imbalance(
                col,
                bins=bins,
                cross_column_names=cross_column_names)
            imbalances.append(imbalance)

        return imbalances

    def balance(self,
            column_name: str,
            cross_column_names: List[str] = [],
            target_upsample: int = None,
            target_downsample: int = 1,
            bins: int = None):

        if not target_upsample:
            target_upsample = self._threshold

        all_cols = cross_column_names + [column_name]
        grouped = self._group_by_columns(all_cols, bins)

        count_grp = grouped.count()
        count_max = count_grp.values.max()
        count_upsample = int(target_upsample*count_max)
        count_downsample = int(target_downsample*count_max)

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
            corr = self.df.corr()
            cols = corr.columns

        if plot_type == "dendogram":
            self._plot_dendogram(corr, cols, figsize=figsize)
        elif plot_type == "matrix":
            self._plot_matrix(corr, cols, figsize=figsize)
        else:
            raise(f"Variable plot_type not valid. Provided: {plot_type}")

        return corr

    


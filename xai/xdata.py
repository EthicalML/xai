import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any

class XData:
    def __init__(self, 
                df: pd.DataFrame = None,
                csv_path: str = None,
                col_names: List[str] = None,
                categorical_cols: List[str] = None,
                string_cols: List[str] = None,
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
                    int((col_max-col_min)/bins))))
                grp = col_bins

            group_list.append(grp)

        grouped = self.df.groupby(group_list)
        return grouped 


    def show_imbalance(self, 
            column_name: str, 
            bins: int = 10, 
            cross_column_names: List[str] = []) -> None:

        all_cols = cross_column_names + [column_name]
        grouped = self._group_by_columns(all_cols, bins)
        grouped_col = grouped[column_name]
        count_grp = grouped_col.count()
        count_max = count_grp.values.max()
        ratios = count_grp/count_max
        imbalances = ratios < self._threshold
        
        cm = plt.cm.get_cmap('RdYlBu_r')
        colors = [cm(1-r/self._threshold/2) if t else cm(0) \
                   for r,t in zip(ratios, imbalances)]
        ax = count_grp.plot.bar(color=colors)
        lp = plt.axhline(self._threshold*count_max, color='r')
        lp.set_label(f"Threshold: {self._threshold*count_max} ({self._threshold*100}%)")
        plt.legend()
        plt.show()
            
        return None

    def show_imbalances(self,
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
            imbalance = self.show_imbalance(
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
        count_upsample = int(target_upsample*count_max+1)
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
        


    


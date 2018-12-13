import pandas as pd
import matplotlib as plt
from typing import List, Any

class XData:
    def __init__(self, 
                label_column: str,
                csv_path: str,
                df: pd.DataFrame = None,
                col_names: List[str] = None,
                categorical_cols: List[str] = None,
                string_cols: List[str] = None,
                **kwargs) -> None:
        
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



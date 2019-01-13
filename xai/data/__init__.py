import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple

PATH = os.path.dirname(os.path.abspath(__file__))

def load_census(return_xy: bool = False
        ) -> Tuple[pd.DataFrame, Optional[np.array]]:
    """
    Load adult census dataset with column as "loan" 
    instead of "target" to use during examples to "automate
    a loan approval process". 

    :Example:

    from xai.data import load_census
    df = load_census()

    :param return_xy: [default: False] pass True if you would like
        to return the data as X, y where X are the input columns and
        y is the target. If nothing (or False) is provided, the default
        return will be the full dataframe.
    :type return_xy: bool
    :returns: Dataframe with full contents OR dataframe with inputs and 
        array with targets.
    :rtype: (pandas.DataFrame, Optional[numpy.array])

    """
    df = pd.read_csv(os.path.join(PATH, "census.csv"), index_col=0)
    if not return_xy:
        return df

    x = df.drop("loan", axis=1)
    y = df["loan"]
    return x, y



from xai.xdata import XData

import pandas as pd
from typing import List, Any, Callable

def predict_func(x,y,m):
    return 

class XModel:
    def __init__(self, 
            xdata: XData,
            trained_model: Any,
            predict_func: Any = lambda x,y,m: m.predict(x)) -> None:
        self._xdata = xdata
        self._model = trained_model



![GitHub](https://img.shields.io/badge/Version-0.1-lightgrey.svg)
![GitHub](https://img.shields.io/badge/Python-3.5 | 3.6-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# XAI - An explainability-first AI library

XAI is a Machine Learning library that is designed with AI explainability in its core. XAI contains various tools that enable for analysis and evaluation of data and models. The XAI library is maintained by [The Institute for Ethical AI & ML](http://ethical.institute/), and it was developed based on the [8 principles for Responsible Machine Learning](http://ethical.institute/principles.html).

## What do we mean by eXplainable AI?

The XAI library is designed to empower machine learning engineers and relevant domain experts to analyse the end-to-end solution and identify discrepancies that may result in sub-optimal performance relative to the objectives required. More concretely, the XAI library is designed using the 3-steps of explainable machine learning, which involve 1) data analysis, 2) model evaluation, and 3) production monitoring. 

The provide a visual overview of these three steps, we can have a look at this in the diagram below:

<img width="100%" src="images/bias.png">


# XAI Quickstart

## Installation

The XAI package is on PyPI. To install you can run:

```
pip install xai
```

Alternatively you can install from source by cloning the repo and running:

```
python setup.py install 
```

## Usage

You can find example usage in the examples folder.

### 1) Data Analysis

```
import xai
from xai.xdata import XData
from xai import xmodel
from xai import xmonitor

xd = XData("path/to/csv/data.csv", "target_col")
#> Loads file, converts numbers to numerical, short str to categorical, long str to text (100 char+)

xd.df.head()
#> Current DF

xd.imbalances(threshold=0.5, plot=True)
#> Show buckets of all features

xd.target_imbalances()
#> Show buckets of all features as breakdown to the target function

xd.partial_imbalances()
#> Show buckets of selected features compared across each other



xd.correlations()
#> Show correlations of all

xd.display.head()
#> DF with unmodified values

print(xd.x_train.shape, xd.y_train.shape)
#> Balanced on 80% by default

print(xd.x_test.shape, xd.y_test.shape)
#> Balanced on 20% by default

xd.train_test_split(balanced_columns=["col1", "col2"])
#> Plot showing how it is balanced on those two classes


```





### 2) Model Evaluation


### 3) Production Monitoring






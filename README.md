![GitHub](https://img.shields.io/badge/Version-0.1-lightgrey.svg)
![GitHub](https://img.shields.io/badge/Python-3.5_|_3.6-blue.svg)
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

#### Load a dataframe to work on
``` python
df = xai.data.census
df.head()
```
<img width="100%" src="images/readme-1.png">

#### View class imbalances for protected columns
``` python
protected_cols = ["gender", "ethnicity", "age"]
ims = xai.show_imbalances(df, protected_cols)
```
<img width="100%" src="images/readme-2.png">

#### View imbalance of one column
``` python
im = xai.show_imbalance(df, "gender")
```
<img width="100%" src="images/readme-3.png">

#### View imbalance of one column intersected with another
``` python
im = xai.show_imbalance(df, "gender", cross=["loan"])
```
<img width="100%" src="images/readme-4.png">

#### Balance the class using upsampling and/or downsampling
``` python
bal_df = xai.balance(df, "gender", cross=["loan"], upsample=1.0)
```
<img width="100%" src="images/readme-5.png">

#### Create a balanced test-train split (should be done pre-balancing)
``` python
x_train, y_train, x_test, y_test = xai.balanced_train_test_split(
            x, y, cross=["gender"], 
            categorical_cols=categorical_cols, min_per_class=300)
```

### 2) Model Evaluation

#### Identify metric imbalances for the whole model
``` python
_= xai.metrics_imbalance(
        x_test, 
        y_test, 
        probabilities)
```

#### Identify metric imbalances for protected columns
``` python
_= xai.metrics_imbalances(
        x_test, 
        y_test, 
        probabilities,
        columns=protected,
        categorical_cols=categorical_cols)
```




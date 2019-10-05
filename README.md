![GitHub](https://img.shields.io/badge/Release-ALPHA-yellow.svg)
![GitHub](https://img.shields.io/badge/Version-0.0.5_ALPHA-lightgrey.svg)
![GitHub](https://img.shields.io/badge/Python-3.5_|_3.6-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# XAI - An eXplainability toolbox for machine learning 

XAI is a Machine Learning library that is designed with AI explainability in its core. XAI contains various tools that enable for analysis and evaluation of data and models. The XAI library is maintained by [The Institute for Ethical AI & ML](http://ethical.institute/), and it was developed based on the [8 principles for Responsible Machine Learning](http://ethical.institute/principles.html).

You can find the documentation at [https://ethicalml.github.io/xai/index.html](https://ethicalml.github.io/xai/index.html). You can also check out our [talk at Tensorflow London](https://www.youtube.com/watch?v=GZpfBhQJ0H4) where the idea was first conceived - the talk also contains an insight on the definitions and principles in this library.

## YouTube video showing how to use XAI to mitigate undesired biases

<table>
  <tr>
    <td width="30%">
        This <a href="https://www.youtube.com/watch?v=vq8mDiDODhc">video of the talk presented at the PyData London 2019 Conference </a> which provides an overview on the motivations for machine learning explainability as well as techniques to introduce explainability and mitigate undesired biases using the XAI Library.
    </td>
    <td width="70%">
        <a href="https://www.youtube.com/watch?v=vq8mDiDODhc"><img src="images/video.jpg"></a>
    </td>
  </tr>
  <tr>
    <td width="30%">
        Do you want to learn about more awesome machine learning explainability tools? Check out our community-built <a href="https://github.com/EthicalML/awesome-machine-learning-operations">"Awesome Machine Learning Production & Operations"</a> list which contains an extensive list of tools for explainability, privacy, orchestration and beyond.
    </td>
    <td width="70%">
        <a href="https://github.com/EthicalML/awesome-machine-learning-operations"><img src="images/mlops-link.png"></a>
    </td>
  </tr>

</table>

# 0.0.5 - ALPHA Version

This library is currently in early stage developments and hence it will be quite unstable due to the fast updates. It is important to bare this in mind if using it in production. 

If you want to see a fully functional demo in action clone this repo and run the <a href="https://github.com/EthicalML/xai/blob/master/examples/XAI%20Example%20Usage.ipynb">Example Jupyter Notebook in the Examples folder</a>.

## What do we mean by eXplainable AI?

We see the challenge of explainability as more than just an algorithmic challenge, which requires a combination of data science best practices with domain-specific knowledge. The XAI library is designed to empower machine learning engineers and relevant domain experts to analyse the end-to-end solution and identify discrepancies that may result in sub-optimal performance relative to the objectives required. More broadly, the XAI library is designed using the 3-steps of explainable machine learning, which involve 1) data analysis, 2) model evaluation, and 3) production monitoring. 

We provide a visual overview of these three steps mentioned above in this diagram:

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

With XAI you can identify imbalances in the data. For this, we will load the census dataset from the XAI library.

``` python
import xai.data
df = xai.data.load_census()
df.head()
```
<img width="100%" src="images/readme-csv-head.jpg">

#### View class imbalances for all categories of one column
``` python
ims = xai.imbalance_plot(df, "gender")
```
<img width="100%" src="images/readme-imbalance-gender.jpg">

#### View imbalances for all categories across multiple columns
``` python
im = xai.show_imbalance(df, "gender", "loan")
```
<img width="100%" src="images/readme-imbalance-multiple.jpg">

#### Balance classes using upsampling and/or downsampling
``` python
bal_df = xai.balance(df, "gender", "loan", upsample=0.8)
```
<img width="100%" src="images/readme-balance-upsample.jpg">

#### Perform custom operations on groups
``` python
groups = xai.group_by_columns(df, ["gender", "loan"])
for group, group_df in groups:    
    print(group) 
    print(group_df["loan"].head(), "\n")
```
<img width="100%" src="images/readme-groups.jpg">

#### Visualise correlations as a matrix
``` python
_ = xai.correlations(df, include_categorical=True, plot_type="matrix")
```
<img width="100%" src="images/readme-correlation-matrix.jpg">

#### Visualise correlations as a hierarchical dendogram
``` python
_ = xai.correlations(df, include_categorical=True)
```
<img width="100%" src="images/readme-correlation-dendogram.jpg">

#### Create a balanced validation and training split dataset
``` python
# Balanced train-test split with minimum 300 examples of 
#     the cross of the target y and the column gender
x_train, y_train, x_test, y_test, train_idx, test_idx = \
    xai.balanced_train_test_split(
            x, y, "gender", 
            min_per_group=300,
            max_per_group=300,
            categorical_cols=categorical_cols)

x_train_display = bal_df[train_idx]
x_test_display = bal_df[test_idx]

print("Total number of examples: ", x_test.shape[0])

df_test = x_test_display.copy()
df_test["loan"] = y_test

_= xai.imbalance_plot(df_test, "gender", "loan", categorical_cols=categorical_cols)
```
<img width="100%" src="images/readme-balance-split.jpg">

### 2) Model Evaluation

We are able to also analyse the interaction between inference results and input features. For this, we will train a single layer deep learning model.

```
model = build_model(proc_df.drop("loan", axis=1))

model.fit(f_in(x_train), y_train, epochs=50, batch_size=512)

probabilities = model.predict(f_in(x_test))
predictions = list((probabilities >= 0.5).astype(int).T[0])
```
<img width="100%" src="images/readme-15.png">

#### Visualise permutation feature importance
``` python
def get_avg(x, y):
    return model.evaluate(f_in(x), y, verbose=0)[1]

imp = xai.feature_importance(x_test, y_test, get_avg)

imp.head()
```
<img width="100%" src="images/readme-6.png">

#### Identify metric imbalances against all test data
``` python
_= xai.metrics_plot(
        y_test, 
        probabilities)
```
<img width="100%" src="images/readme-metrics-plot.jpg">

#### Identify metric imbalances across a specific column
``` python
_ = xai.metrics_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=["gender"],
    categorical_cols=categorical_cols)
```
<img width="100%" src="images/readme-metrics-column.jpg">

#### Identify metric imbalances across multiple columns
``` python
_ = xai.metrics_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=["gender", "ethnicity"],
    categorical_cols=categorical_cols)
```
<img width="100%" src="images/readme-metrics-multiple.jpg">

#### Draw confusion matrix
``` python
xai.confusion_matrix_plot(y_test, pred)
```
<img width="100%" src="images/readme-confusion-matrix.jpg">

#### Visualise the ROC curve against all test data
``` python
_ = xai.roc_plot(y_test, probabilities)
```
<img width="100%" src="images/readme-9.png">

#### Visualise the ROC curves grouped by a protected column
``` python
protected = ["gender", "ethnicity", "age"]
_ = [xai.roc_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=[p],
    categorical_cols=categorical_cols) for p in protected]
```
<img width="100%" src="images/readme-10.png">

#### Visualise accuracy grouped by probability buckets
``` python
d = xai.smile_imbalance(
    y_test, 
    probabilities)
```
<img width="100%" src="images/readme-12.png">

#### Visualise statistical metrics grouped by probability buckets
``` python
d = xai.smile_imbalance(
    y_test, 
    probabilities,
    display_breakdown=True)
```
<img width="100%" src="images/readme-13.png">

#### Visualise benefits of adding manual review on probability thresholds
``` python
d = xai.smile_imbalance(
    y_test, 
    probabilities,
    bins=9,
    threshold=0.75,
    manual_review=0.375,
    display_breakdown=False)
```
<img width="100%" src="images/readme-14.png">





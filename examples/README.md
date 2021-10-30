```python
import sys, os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# Use below for charts in dark jupyter theme

THEME_DARK = False

if THEME_DARK:
    # This is used if Jupyter Theme dark is enabled. 
    # The theme chosen can be activated with jupyter theme as follows:
    # >>> jt -t oceans16 -T -nfs 115 -cellw 98% -N  -kl -ofs 11 -altmd
    font_size = '20.0'
    dark_theme_config = {
        "ytick.color" : "w",
        "xtick.color" : "w",
        "text.color": "white",
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size, 
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size, 
        'legend.fontsize': font_size, 
        'figure.titlesize': font_size,
        'figure.figsize': [20, 7],
        'figure.facecolor': "#384151",
        'legend.facecolor': "#384151",
        "axes.labelcolor" : "w",
        "axes.edgecolor" : "w"
    }
    plt.rcParams.update(dark_theme_config)

sys.path.append("..")

import xai
import xai.data
```


```python
csv_path = 'data/adult.data'
categorical_cols = ["gender", "workclass", "education", "education-num", "marital-status",
                   "occupation", "relationship", "ethnicity", "loan"]
csv_columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "ethnicity", "gender", "capital-gain", "capital-loss",
                   "hours-per-week", "loan"]
```


```python
df = xai.data.load_census()
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>ethnicity</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>loan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
target = "loan"
protected = ["ethnicity", "gender", "age"]
```


```python
df_groups = xai.imbalance_plot(df, "gender", categorical_cols=categorical_cols)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_4_0.png)
    



```python
groups = xai.imbalance_plot(df, "gender", "loan", categorical_cols=categorical_cols)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_5_0.png)
    



```python
bal_df = xai.balance(df, "gender", "loan", upsample=0.8, categorical_cols=categorical_cols)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_6_0.png)
    



```python
groups = xai.group_by_columns(df, ["gender", "loan"], categorical_cols=categorical_cols)
for group, group_df in groups:
    print(group)
    print(group_df["loan"].head(), "\n")
```

    (' Female', ' <=50K')
    4      <=50K
    5      <=50K
    6      <=50K
    12     <=50K
    21     <=50K
    Name: loan, dtype: object 
    
    (' Female', ' >50K')
    8      >50K
    19     >50K
    52     >50K
    67     >50K
    84     >50K
    Name: loan, dtype: object 
    
    (' Male', ' <=50K')
    0      <=50K
    1      <=50K
    2      <=50K
    3      <=50K
    13     <=50K
    Name: loan, dtype: object 
    
    (' Male', ' >50K')
    7      >50K
    9      >50K
    10     >50K
    11     >50K
    14     >50K
    Name: loan, dtype: object 
    



```python
_ = xai.correlations(df, include_categorical=True, plot_type="matrix")
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_8_0.png)
    



```python
_ = xai.correlations(df, include_categorical=True)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_9_0.png)
    



```python
proc_df = xai.normalize_numeric(bal_df)
proc_df = xai.convert_categories(proc_df)
x = proc_df.drop("loan", axis=1)
y = proc_df["loan"]

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

    Total number of examples:  1200



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_10_1.png)
    



```python
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, roc_curve, auc

from tensorflow.keras.layers import Input, Dense, Flatten, \
    Concatenate, concatenate, Dropout, Lambda, Embedding
from tensorflow.keras.models import Model, Sequential

def build_model(X):
    input_els = []
    encoded_els = []
    dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
    for k,dtype in dtypes:
        input_els.append(Input(shape=(1,)))
        if dtype == "int8":
            e = Flatten()(Embedding(X[k].max()+1, 1)(input_els[-1]))
        else:
            e = input_els[-1]
        encoded_els.append(e)
    encoded_els = concatenate(encoded_els)

    layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
    out = Dense(1, activation='sigmoid')(layer1)

    # train model
    model = Model(inputs=input_els, outputs=[out])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model


def f_in(X, m=None):
    """Preprocess input so it can be provided to a function"""
    if m:
        return [X.iloc[:m,i] for i in range(X.shape[1])]
    else:
        return [X.iloc[:,i] for i in range(X.shape[1])]

def f_out(probs, threshold=0.5):
    """Convert probabilities into classes"""
    return list((probs >= threshold).astype(int).T[0])

```


```python
model = build_model(x_train)

model.fit(f_in(x_train), y_train, epochs=50, batch_size=512)
```

    Epoch 1/50
    99/99 [==============================] - 1s 3ms/step - loss: 0.6227 - accuracy: 0.6459
    Epoch 2/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.4600 - accuracy: 0.7812
    Epoch 3/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3968 - accuracy: 0.8153
    Epoch 4/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3789 - accuracy: 0.8215
    Epoch 5/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3751 - accuracy: 0.8237
    Epoch 6/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3771 - accuracy: 0.8235
    Epoch 7/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3730 - accuracy: 0.8254
    Epoch 8/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3675 - accuracy: 0.8312
    Epoch 9/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3685 - accuracy: 0.8281
    Epoch 10/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3620 - accuracy: 0.8313
    Epoch 11/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3687 - accuracy: 0.8297
    Epoch 12/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3698 - accuracy: 0.8292
    Epoch 13/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3666 - accuracy: 0.8285
    Epoch 14/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3649 - accuracy: 0.8305
    Epoch 15/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3628 - accuracy: 0.8326
    Epoch 16/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3669 - accuracy: 0.8306
    Epoch 17/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3587 - accuracy: 0.8347
    Epoch 18/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3639 - accuracy: 0.8306
    Epoch 19/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3618 - accuracy: 0.8335
    Epoch 20/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3628 - accuracy: 0.8315
    Epoch 21/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3641 - accuracy: 0.8325
    Epoch 22/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3634 - accuracy: 0.8310
    Epoch 23/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3626 - accuracy: 0.8293
    Epoch 24/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3659 - accuracy: 0.8298
    Epoch 25/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3607 - accuracy: 0.8333
    Epoch 26/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3600 - accuracy: 0.8321
    Epoch 27/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3650 - accuracy: 0.8296
    Epoch 28/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3626 - accuracy: 0.8317
    Epoch 29/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3654 - accuracy: 0.8310
    Epoch 30/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3659 - accuracy: 0.8322
    Epoch 31/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3716 - accuracy: 0.8278
    Epoch 32/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3631 - accuracy: 0.8326
    Epoch 33/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3669 - accuracy: 0.8312
    Epoch 34/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3604 - accuracy: 0.8325
    Epoch 35/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3625 - accuracy: 0.8318
    Epoch 36/50
    99/99 [==============================] - 0s 2ms/step - loss: 0.3605 - accuracy: 0.8326
    Epoch 37/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3595 - accuracy: 0.8334
    Epoch 38/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3653 - accuracy: 0.8316
    Epoch 39/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3591 - accuracy: 0.8350
    Epoch 40/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3602 - accuracy: 0.8337
    Epoch 41/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3617 - accuracy: 0.8316
    Epoch 42/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3624 - accuracy: 0.8320
    Epoch 43/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3624 - accuracy: 0.8328
    Epoch 44/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3602 - accuracy: 0.8326
    Epoch 45/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3610 - accuracy: 0.8337
    Epoch 46/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3626 - accuracy: 0.8323
    Epoch 47/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3634 - accuracy: 0.8326
    Epoch 48/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3614 - accuracy: 0.8328
    Epoch 49/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3610 - accuracy: 0.8332
    Epoch 50/50
    99/99 [==============================] - 0s 3ms/step - loss: 0.3590 - accuracy: 0.8332





    <tensorflow.python.keras.callbacks.History at 0x7f1ee46da710>




```python
score = model.evaluate(f_in(x_test), y_test, verbose=1)
print("Error %.4f: " % score[0])
print("Accuracy %.4f: " % (score[1]*100))
```

    38/38 [==============================] - 0s 1ms/step - loss: 0.3630 - accuracy: 0.8292
    Error 0.3630: 
    Accuracy 82.9167: 



```python
probabilities = model.predict(f_in(x_test))
pred = f_out(probabilities)
```


```python
_= xai.metrics_plot(
        y_test, 
        probabilities)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_15_0.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>ethnicity</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>loan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = xai.metrics_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=["gender", "ethnicity"],
    categorical_cols=categorical_cols)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_17_0.png)
    



```python
_ = [xai.metrics_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=[p],
    categorical_cols=categorical_cols) for p in protected]
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_18_0.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_18_1.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_18_2.png)
    



```python
xai.confusion_matrix_plot(y_test, pred)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_19_0.png)
    



```python
xai.confusion_matrix_plot(y_test, pred, scaled=False)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_20_0.png)
    



```python
_ = xai.roc_plot(y_test, probabilities)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_21_0.png)
    



```python
_ = [xai.roc_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=[p],
    categorical_cols=categorical_cols) for p in protected]
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_22_0.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_22_1.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_22_2.png)
    



```python
_= xai.pr_plot(y_test, probabilities)
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_23_0.png)
    



```python
_ = [xai.pr_plot(
    y_test, 
    probabilities, 
    df=x_test_display, 
    cross_cols=[p],
    categorical_cols=categorical_cols) for p in protected]
```


    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_24_0.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_24_1.png)
    



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_24_2.png)
    



```python
d = xai.smile_imbalance(
    y_test, 
    probabilities)
```

    /home/alejandro/miniconda3/lib/python3.7/site-packages/pandas/core/indexing.py:671: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    WARNING:root:No categorical_cols passed so inferred using np.object, np.int8 and np.bool: Index(['target', 'manual-review'], dtype='object'). If you see an error these are not correct, please provide them as a string array as: categorical_cols=['col1', 'col2', ...]



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_25_1.png)
    



```python
d[["correct", "incorrect"]].sum().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1effc41fd0>




    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_26_1.png)
    



```python
d = xai.smile_imbalance(
    y_test, 
    probabilities,
    threshold=0.75,
    display_breakdown=True)
```

    WARNING:root:No categorical_cols passed so inferred using np.object, np.int8 and np.bool: Index(['target', 'manual-review'], dtype='object'). If you see an error these are not correct, please provide them as a string array as: categorical_cols=['col1', 'col2', ...]



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_27_1.png)
    



```python
display_bars = ["true-positives", "true-negatives", 
                "false-positives", "false-negatives"]
d[display_bars].sum().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1effecd990>




    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_28_1.png)
    



```python
d = xai.smile_imbalance(
    y_test, 
    probabilities,
    bins=9,
    threshold=0.75,
    manual_review=0.00001,
    display_breakdown=False)
```

    WARNING:root:No categorical_cols passed so inferred using np.object, np.int8 and np.bool: Index(['target', 'manual-review'], dtype='object'). If you see an error these are not correct, please provide them as a string array as: categorical_cols=['col1', 'col2', ...]



    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_29_1.png)
    



```python
d[["correct", "incorrect", "manual-review"]].sum().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1ec01c9850>




    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_30_1.png)
    



```python
def get_avg(x, y):
    return model.evaluate(f_in(x), y, verbose=0)[1]

imp = xai.feature_importance(x_test, y_test, get_avg)

imp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>ethnicity</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01825</td>
      <td>0.002167</td>
      <td>0.000833</td>
      <td>0.046</td>
      <td>0.065667</td>
      <td>0.019083</td>
      <td>0.02425</td>
      <td>0.00275</td>
      <td>0.000833</td>
      <td>0.05075</td>
      <td>0.007833</td>
      <td>0.014417</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](XAI%20Tabular%20Data%20Example%20Usage_files/XAI%20Tabular%20Data%20Example%20Usage_31_1.png)
    



```python

```

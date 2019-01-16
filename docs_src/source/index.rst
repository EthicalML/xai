.. xai - WordCount Python documentation master file, created by
   sphinx-quickstart on Fri Jun 23 15:52:18 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the XAI docs - eXplainable machine learning
===========================================================

.. toctree::
   :maxdepth: 4 


Welcome to the ``xai`` documentation. Here you will the installation guide, the quick start guide, and the detailed docstrings code documentation of the xai library.

The documentation is currently under construction - for the meantime you can check out the main Github repository for the code (https://github.com/EthicalML/xai).

About
================

XAI is a Machine Learning library that is designed with AI explainability in its core. XAI contains various tools that enable for analysis and evaluation of data and models. The XAI library is maintained by `The Institute for Ethical AI & ML <http://ethical.institute/>`_, and it was developed based on the `8 principles for Responsible Machine Learning <http://ethical.institute/principles.html>`_.

You can find the documentation at https://ethicalml.github.io/xai/index.html. You can also check out our `talk at Tensorflow London <https://www.youtube.com/watch?v=GZpfBhQJ0H4>`_ where the idea was first conceived - the talk also contains an insight on the definitions and principles in this library.

0.0.4 - ALPHA Version
------------------------

This library is currently in early stage developments and hence it will be quite unstable due to the fast updates. It is important to bare this in mind if using it in production. 

What do we mean by eXplainable AI?
---------------------------------------

We see the challenge of explainability as more than just an algorithmic challenge, which requires a combination of data science best practices with domain-specific knowledge. The XAI library is designed to empower machine learning engineers and relevant domain experts to analyse the end-to-end solution and identify discrepancies that may result in sub-optimal performance relative to the objectives required. More broadly, the XAI library is designed using the 3-steps of explainable machine learning, which involve 1) data analysis, 2) model evaluation, and 3) production monitoring. 

We provide a visual overview of these three steps mentioned above in this diagram:

.. image:: _static/bias.png

XAI Quickstart
====================================

Installation
------------------

The XAI package is on PyPI. To install you can run:

.. parsed-literal::

   pip install xai

Alternatively you can install from source by cloning the repo and running:

.. parsed-literal::

   python setup.py install 

Usage
---------

You can find example usage in the examples folder.

1) Data Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^


With XAI you can identify imbalances in the data. For this, we will load the census dataset from the XAI library.

.. parsed-literal::

   import xai.data
   df = xai.data.load_census()
   df.head()

.. image:: _static/readme-1.png

View class imbalances for protected columns
""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. parsed-literal::

   protected_cols = ["gender", "ethnicity", "age"]
   ims = xai.show_imbalances(df, protected_cols)

.. image:: _static/readme-2.png

View imbalance of one column
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   im = xai.show_imbalance(df, "gender")

.. image:: _static/readme-3.png

View imbalance of one column intersected with another
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   im = xai.show_imbalance(df, "gender", cross=["loan"])

.. image:: _static/readme-4.png

Balance the class using upsampling and/or downsampling
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::
   bal_df = xai.balance(df, "gender", cross=["loan"], upsample=1.0)

.. image:: _static/readme-5.png

Create a balanced test-train split (should be done pre-balancing)
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   # Balanced train-test split with minimum 300 examples of 
   # the cross of the target y and the column gender
   x_train, y_train, x_test, y_test = xai.balanced_train_test_split(
               x, y, cross=["gender"], 
               categorical_cols=categorical_cols, min_per_class=300)

   # Visualise the imbalances of gender and the target 
   df_test = x_test.copy()
   df_test["loan"] = y_test
   _= xai.show_imbalance(df_test, "gender", cross=["loan"], categorical_cols=categorical_cols)

.. image:: _static/readme-16.png

2) Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

We are able to also analyse the interaction between inference results and input features. For this, we will train a single layer deep learning model.

.. parsed-literal::

   model = build_model(proc_df.drop("loan", axis=1))
   model.fit(f_in(x_train), y_train, epochs=50, batch_size=512)

   probabilities = model.predict(f_in(x_test))
   predictions = list((probabilities >= 0.5).astype(int).T[0])

.. image:: _static/readme-15.png

Visualise permutation feature importance
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   def get_avg(x, y):
       return model.evaluate(f_in(x), y, verbose=0)[1]

   imp = xai.feature_importance(x_test, y_test, get_avg)

   imp.head()

.. image:: _static/readme-6.png

Identify metric imbalances against all test data
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   _= xai.metrics_imbalance(
           x_test, 
           y_test, 
           probabilities)

.. image:: _static/readme-7.png

Identify metric imbalances grouped by protected columns
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::
   _= xai.metrics_imbalances(
           x_test, 
           y_test, 
           probabilities,
           columns=protected,
           categorical_cols=categorical_cols)

.. image:: _static/readme-8.png

Visualise the ROC curve against all test data
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   _= xai.roc_imbalance(
          x_test, 
          y_test, 
          probabilities)

.. image:: _static/readme-9.png

Visualise the ROC curves grouped by protected columns
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::
   _= xai.roc_imbalances(
       x_test, 
       y_test, 
       probabilities, 
       columns=protected,
       categorical_cols=categorical_cols)

.. image:: _static/readme-10.png

Visualise the precision-recall curve by protected columns
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   _= xai.pr_imbalances(
       x_test, 
       y_test, 
       probabilities, 
       columns=protected,
       categorical_cols=categorical_cols)
       
.. image:: _static/readme-11.png

Visualise accuracy grouped by probability buckets
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   d = xai.smile_imbalance(
       y_test, 
       probabilities)
       
.. image:: _static/readme-12.png

Visualise statistical metrics grouped by probability buckets
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   d = xai.smile_imbalance(
       y_test, 
       probabilities,
       display_breakdown=True)
       
.. image:: _static/readme-13.png

Visualise benefits of adding manual review on probability thresholds
""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. parsed-literal::

   d = xai.smile_imbalance(
       y_test, 
       probabilities,
       bins=9,
       threshold=0.75,
       manual_review=0.375,
       display_breakdown=False)
       
.. image:: _static/readme-14.png



xai Python Docstrings
===========================


Submodules
------------

xai\.data module
-----------------

.. automodule:: xai.data
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
-----------------

.. automodule:: xai
    :members:
    :undoc-members:
    :show-inheritance:


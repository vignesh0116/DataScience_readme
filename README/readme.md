---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.13
  nbformat: 4
  nbformat_minor: 5
---

<div class="cell markdown">

# Default of credit card clients

</div>

<div class="cell markdown">

### Table of Contents

1.  Loading the data
2.  Preprocessing the data
3.  Explore features or charecteristics to predict default of credit
    card clients
4.  Develop prediction models
5.  Evaluate and refine prediction models

</div>

<div class="cell markdown" tags="[]">

### Project Overview

This Project aimed at the case of customers default payments in Taiwan
and compares the predictive accuracy of probability of default. From the
perspective of risk management, the result of predictive accuracy of the
estimated probability of default will be more valuable than the binary
result of classification - credible or not credible clients.

I worked with the project in Default of credit card clients. I focused
on predicting the Default list of customer. I followed a process of
problem definition, gathering data, preparing data, explanatory data
analysis, coming up with a data model, validating the model, and
optimizing the model further.

Let's take a look at the steps

</div>

<div class="cell markdown" tags="[]">

### Data Science Steps

1.  Problem Definition: What factors determined whether someone survived
    a disaster? Using passenger data, we were able to identify certain
    groups of people who were more likely to survive.
2.  Data Gathering: Kaggle provided the input data on their website.
3.  Data Preparation: I prepared the data by analyzing data points that
    were missing or outliers.
4.  EDA (Exploratory Data Analysis): If you input garbage data into a
    system, you'll get garbage output. Therefore, it is important to use
    descriptive and graphical statistics to look for patterns,
    correlations and comparisons in the dataset. In this step, I
    analyzed the data to make sure it was understandable.
5.  Data Modeling: It is important to know when to select a model. If we
    choose the wrong model for a particular use case, all other steps
    become pointless.
6.  Validate Model: After training the model, I checked its performance
    and looked for any issues with overfitting or underfitting.
7.  Optimize Model: Using techniques like hyperparameter optimization, I
    worked on making the model better.

</div>

<div class="cell markdown" tags="[]">

### Step 1: Problem Definition

This research aimed at the case of customers default payments in Taiwan
and compares the predictive accuracy of probability of default among
different models methods.

</div>

<div class="cell markdown">

### Step 2: Data Gathering

The dataset can be found on found on [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

</div>

<div class="cell markdown" tags="[]">

### Step 3: Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.

</div>

<div class="cell markdown">

#### 3.1 Import Libraries

</div>

<div class="cell code" execution_count="62">

``` python
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

import pickle

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
```

<div class="output stream stdout">

    Python version: 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
    pandas version: 1.4.4
    matplotlib version: 3.5.2
    NumPy version: 1.21.5
    SciPy version: 1.9.1
    IPython version: 7.31.1
    scikit-learn version: 1.2.2
    -------------------------

</div>

</div>

<div class="cell code" execution_count="2">

``` python
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import roc_auc_score

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
```

</div>

<div class="cell markdown">

#### 3.2 Pre-view of the Data

</div>

<div class="cell markdown">

The default payment next month variable is the outcome or dependent
variable. The datatype is 1 if the customer default and 0 if they did
not default. The rest of the variables are independent variables. Most
variable names are self explanatory but a couple may be worth
mentioning. The LIMIT_BAL represents Amount of the given credit (NT
dollar): it includes both the individual consumer credit and his/her
family (supplementary) credit.Pay_6 to Pay_0 is the History of past
payment. We tracked the past monthly payment records (from April to
September, 2005).similarly for BILL_AMT & PAY_AMT columns respectively.

</div>

<div class="cell markdown">

![image.png](a46ca2ee-ace7-493d-bc89-a96123207b49.png)

</div>

<div class="cell markdown">

![image.png](29824355-38b5-4e02-a987-7af61b14a351.png)

</div>

<div class="cell markdown">

[image.png](attachment:f813f3f5-b78b-4efb-8aae-72761f96817c.png)

</div>

<div class="cell markdown">

Check for missing values

</div>

<div class="cell markdown">

![image.png](a5728728-e35e-40fd-9f36-afb9f5072d41.png)

</div>

<div class="cell markdown">

Ratio between Male and female (1 = male; 2 = female)

![image.png](f2eb6ec4-2b3d-4e6d-a4ae-50bd834eed25.png)

</div>

<div class="cell markdown">

#### 3.3 Data Pre-processing:

</div>

<div class="cell markdown">

##### Rename Columns by month

</div>

<div class="cell markdown">

PAY_0 to PAY_6 history of past payment. We tracked the past monthly
payment records (from April to September, 2005), similarly for BILL_AMT
& PAY_AMT columns respectively.

To change the columns accordingly to respective month to get clear idea
of what the features does.

-   PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 changed to PAY_September,
    PAY_August, PAY_July, PAY_June, PAY_May, PAY_April
-   BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, to
    BILL_AMT_September, BILL_AMT_August, BILL_AMT_July, BILL_AMT_June,
    BILL_AMT_May, BILL_AMT_April
-   PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, to
    PAY_AMT_September, PAY_AMT_August, PAY_AMT_July, PAY_AMT_June,
    PAY_AMT_May, PAY_AMT_April
-   Rename the Target column to avoid space/column length

</div>

<div class="cell markdown">

![image.png](900ee1d7-27a6-47b9-908a-5d56cd3ab489.png)

</div>

<div class="cell markdown">

![image.png](d144beb7-6c93-4b2d-9457-36f935414363.png)

</div>

<div class="cell markdown">

Rounding the value to four categories to understand easily

</div>

<div class="cell markdown">

![image.png](01b08bff-6980-4418-ae1a-0b24229b2e85.png)

</div>

<div class="cell markdown">

### Step 4: Explanatory Data Analysis (EDA)

</div>

<div class="cell markdown">

-   Below graph shows Male mosty get defaulted than Female
-   highschool gets defaulted than other categories
-   Married person get defaulted than single

</div>

<div class="cell markdown">

![image.png](659879ca-349e-407c-9eb7-106987730533.png)

</div>

<div class="cell markdown">

Male at all age bins get defaulted than Female as we seen above married
male gets default than married female
![image.png](4b7483a7-54fc-4e04-9293-169c068718c9.png)

</div>

<div class="cell markdown">

As we seen below low limit score mostly to get default

![image.png](899a38be-d393-4bdd-9d4e-9001ed0a4217.png)

</div>

<div class="cell markdown">

Pay_september status \> 2 mostly gets defaulted

![image.png](64e3bca0-d610-4c5c-8d4c-6bd430d6ebf3.png)

</div>

<div class="cell markdown">

Predicting the Good feature for our model using ANOVA for All features
vs Target. Pay_status features will impact more thaan others lets
compare later with feature importance

![image.png](b6cbf69b-6a6c-430b-a678-2022588d5901.png)

</div>

<div class="cell markdown">

Top 25% feature are listed below to get good output

![image.png](d85e6a1a-a66b-4ef0-aca9-f9c36c374d13.png)

</div>

<div class="cell code">

``` python
```

</div>

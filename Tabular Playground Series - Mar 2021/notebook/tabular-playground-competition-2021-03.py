#!/usr/bin/env python
# -*- coding: utf-8 -*-
# In[46]:
import os
import uuid
from pathlib import Path
import pandas_profiling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# In[7]:
input_path = Path('../data')
# In[8]:
# https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
np.set_printoptions(suppress=True)  # Print floating point numbers using fixed point notation.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
# Print out the full DataFrame repr for wide DataFrames across multiple lines.
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_columns', 500)  # Set to None for unlimited number of output rows.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.max_rows', 500)  # Set to None for unlimited number of output rows.
pd.set_option('display.width', 120)  # Width of the display in characters.
# # Read in the data files
# In[9]:
train = pd.read_csv(input_path / 'train.csv.gz', index_col='id')
print(train.shape)
display(train.head())
# In[10]:
test = pd.read_csv(input_path / 'test.csv.gz', index_col='id')
print(test.shape)
display(test.head())
# In[11]:
submission = pd.read_csv(input_path / 'sample_submission.csv.gz', index_col='id')
display(submission.head())
# # Balance of target
# In[12]:
train['target'].value_counts(normalize=True)
# Not 1:1 but not very imbalanced.
# # EDA
# In[13]:
# profile = pandas_profiling.ProfileReport(train, explorative=True)
# In[14]:
# profile.to_notebook_iframe()
# Observations:
#
# * No missing cells, duplicate rows, or outliers.
# * cat5, cat7, cat8, cat10 have high cardinality
# * cat1, cat2, cat3, cat5, cat6, cat9 have medium cardinality
# * cat0, cat11, cat12, cat13, cat14 have only two categories. cat0 and cat14 are more balanced.
# * cat15, cat16, cat17, cat18 have 4 but could be made into binary unless the individual categories have high correlation with target.
# ## EDA of Continuous Variables
# In[15]:
cont_vars = [c for c in train.columns if c.startswith('cont')]
cont_vars
# In[16]:
# for cont_var in cont_vars:
#     sns.histplot(
#         x=cont_var,
#         data=train,
#         hue='target',
#     );
#     plt.show();
# In[17]:
# pd.plotting.scatter_matrix(train, alpha=0.2);
# In[18]:
# corrDf = train[cont_vars + ['target']].corr().abs()
# for col in corrDf.columns:
#     corrDf.loc[col, col] = 0
# idxmax = corrDf.idxmax(axis='columns')
# max_ = corrDf.max(axis='columns')
# (
#     pd.DataFrame(
#         index=corrDf.index,
#         data={
#             'idxmax': idxmax,
#             'max': max_,
#         },
#     )
#     .sort_values(
#         by=[
#             'max',
#         ],
#         ascending=False,
#     )
# )
# Observations:
#
# * Cont1 and Cont2 are highly dependent on each other! Better to remove one of them, or take the difference between them as a feature.
# * Cont0 and Cont10 are also highly dependent on each other, as are count7 and count10.
#
# ## EDA of Categorical Variables
# In[19]:
cat_vars = [c for c in train.columns if c.startswith('cat')]
cat_vars
# In[20]:
for cat_var in cat_vars:
    train[cat_var] = train[cat_var].astype('category')
# In[21]:
# for cat_var in cat_vars:
#     sns.catplot(x=cat_var, y="target", data=train, kind="bar",);
#     plt.show();
# # Transformations
# ## Transform Continuous Variables
# In[22]:
def subtractCorrelated(df):
    df['cont1m2'] = df['cont1'] - df['cont2']
    df['cont1m10'] = df['cont1'] - df['cont10']
    df['cont0m7'] = df['cont0'] - df['cont7']
    df['cont1m8'] = df['cont1'] - df['cont8']
    df = (
        df
        .drop(
            columns=[
                'cont2',
                'cont10',
                'cont7',
                'cont8',
            ],
        )
    )
    return df
# In[23]:
# train = subtractCorrelated(train)
# test = subtractCorrelated(test)
# In[24]:
train.shape, test.shape
# In[25]:
# corrDf = train[cont_vars + ['target']].corr().abs()
# for col in corrDf.columns:
#     corrDf.loc[col, col] = 0
# idxmax = corrDf.idxmax(axis='columns')
# max_ = corrDf.max(axis='columns')
# display(
#     pd.DataFrame(
#         index=corrDf.index,
#         data={
#             'idxmax': idxmax,
#             'max': max_,
#         },
#     )
#     .sort_values(
#         by=[
#             'max',
#         ],
#         ascending=False,
#     )
# )
# display(corrDf['target'])
# ## Transform Categorical Variables
# In[26]:
def manuallyHandleCategorical(df):
    # cat15, cat16, cat17, cat18 have 4 but could be made into binary.
    mask = df['cat15'].isin(['B', 'D'])
    df.loc[~mask, 'cat15'] = 'B'  # Replace others with mode.
    mask = df['cat16'].isin(['B', 'D'])
    df.loc[~mask, 'cat16'] = 'D'  # Replace others with mode.
    mask = df['cat17'].isin(['B', 'D'])
    df.loc[~mask, 'cat17'] = 'D'  # Replace others with mode.
    mask = df['cat18'].isin(['B', 'D'])
    df.loc[~mask, 'cat18'] = 'D'  # Replace others with mode.

    # cat1, cat2, cat3, cat5, cat6, cat9 have medium cardinality
    col = 'cat1'
    s = ['I', 'F', 'K', 'L', 'H']
    mode = 'I'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat2'
    s = ['A', 'C', 'D', 'G']
    mode = 'A'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat3'
    s = ['A', 'B', 'C']
    mode = 'A'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat5'
    s = ['BI', 'AB']
    mode = 'BI'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat6'
    s = ['A', 'C', 'E']
    mode = 'A'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat9'
    s = ['A', 'C', 'E']
    mode = 'A'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat7'
    s = ['AH', 'E', 'AS', 'J', 'AN', 'U']
    mode = 'AH'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat8'
    s = ['BM', 'AE', 'AX', 'Y', 'H']
    mode = 'BM'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
    col = 'cat10'
    s = ['DJ', 'HK', 'DP', 'GS']
    mode = 'DJ'
    mask = df[col].isin(s)
    df.loc[~mask, col] = mode  # Replace others with mode.
# In[27]:
# train = manuallyHandleCategorical(train)
# test = manuallyHandleCategorical(test)
# In[28]:
train.shape, test.shape
# # Train
# ## Pull out the target, and make a validation split
# In[29]:
target = train.pop('target')
# In[30]:
train.shape, test.shape
# In[31]:
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25)
# In[32]:
def getOHE():
    return OneHotEncoder(sparse=False)
# In[33]:
def getLE():
    return LabelEncoder()
# In[34]:
enc = getLE()
# enc = getOHE()
# In[35]:
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()
for c in X_train_enc[cat_vars].columns:
    enc.fit(list(X_train_enc[c].values) + list(X_test_enc[c].values))
    X_train_enc[c] = enc.transform(X_train_enc[c].values)
    X_test_enc[c] = enc.transform(X_test_enc[c].values)
# In[36]:
X_train.shape, X_train_enc.shape, X_test.shape, X_test_enc.shape
# In[37]:
X_train_enc.head()
# In[38]:
def getRFC():
    return RandomForestClassifier(n_estimators=200, max_depth=7, n_jobs=-1, verbose=1)
# In[39]:
def getXgb():
    return xgb.XGBClassifier(
        n_jobs=-1,
        random_state=0,
        use_label_encoder=False,
    )
# In[40]:
def getRsXgb():
    params = {
        'min_child_weight': [1, 4],
        'gamma': [0, 0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 6, 8],
    }

    clf = xgb.XGBClassifier(
        n_jobs=-1,
        random_state=0,
        use_label_encoder=False,
    )

    return RandomizedSearchCV(  # FIXME: Try new one?
        clf,
        param_distributions=params,
        scoring='roc_auc',  # FIXME: Try f1, f1_weighted?
        random_state=0,
        n_jobs=-1,
        verbose=2,
    )
# In[41]:
get_ipython().run_cell_magic('time', '', "# clf = getRFC()\nclf = getRsXgb()\nclf.fit(X_train_enc, y_train)\ny_pred = clf.predict_proba(X_test_enc)[:, 1] # This grabs the positive class prediction\nscore = roc_auc_score(y_test, y_pred)\nprint(f'{score:0.5f} v 0.88962 reference value')  # New value ?!")
# ## Let's take a look at how the model predicted the various classes
#
# The graph below shows that the model does well with most of the negative observations, but struggles with many of the positive observations.
# In[42]:
plt.figure(figsize=(8,4))
plt.hist(y_pred[np.where(y_test == 0)], bins=100, alpha=0.75, label='neg class')
plt.hist(y_pred[np.where(y_test == 1)], bins=100, alpha=0.75, label='pos class')
plt.legend()
plt.show()
# # Train it on all the data and create submission
# In[43]:
# enc = getOHE()
enc = getLE()
# In[44]:
train_enc = train.copy()
test_enc = test.copy()
for c in train_enc[cat_vars].columns:
    enc.fit(list(train_enc[c].values) + list(test_enc[c].values))
    train_enc[c] = enc.transform(train_enc[c].values)
    test_enc[c] = enc.transform(test_enc[c].values)
# In[45]:
get_ipython().run_cell_magic('time', '', "# clf = getRFC()\nclf = getRsXgb()\nclf.fit(train_enc, target)\nsubmission['target'] = clf.predict_proba(test_enc)[:, 1]")
# In[48]:
filename = f'submission_{uuid.uuid4()}.csv'
submission.to_csv(input_path / filename)
filename
# In[ ]:

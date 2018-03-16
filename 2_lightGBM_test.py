import lightgbm as lgb
import zipfile
import sys
import time
import datetime
import os
import pandas as pd
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
import my_utils


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def logloss_baseline(y_train, y_val):
    print('\n##### baseline #####')
    mean_cvr_train = y_train.mean()
    pred_by_train = len(y_val) * [mean_cvr_train]
    print('mean_cvr_train:', mean_cvr_train)
    print('pred by mean_cvr_train:', logloss(act=y_val, pred=pred_by_train))
    mean_cvr_val = y_val.mean()
    pred_by_val = len(y_val) * [mean_cvr_val]
    print('mean_cvr_val:', mean_cvr_val)
    print('pred by mean_cvr_val:', logloss(act=y_val, pred=pred_by_val))
    print('######################\n')


def get_data_and_label(df, feature_cols, label_col):
    x = df[feature_cols].values
    y = df[label_col].values
    return x, y


lgb_clf = lgb.LGBMClassifier(
            # max_depth=param1,
            learning_rate=0.03,
            num_leaves=25,
            n_estimators=1500,
            min_child_weight=5,
            nthread=6,
            )

df = pd.read_csv("../data/df_concat.csv")
df.fillna(0, inplace=True)

feature_cols = my_utils.feature_cols

label_col = 'is_trade'

df_train = my_utils.select_range_by_day(df, 18, 23)
df_val = my_utils.select_range_by_day(df, 24, 24)
X_train, y_train = get_data_and_label(df_train, feature_cols, label_col)
X_val, y_val = get_data_and_label(df_val, feature_cols, label_col)

logloss_baseline(y_train, y_val)

model = lgb_clf
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50,
          verbose=50,
        )
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


def logit(x):
    return np.log(x/(1-x))


def logistic(x):
    return 1/(1+np.exp(-x))


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

df = pd.read_csv("../data/df_merge.csv")
print(df.columns)
df.fillna(0, inplace=True)

feature_cols = my_utils.feature_cols

feature_cols = [
        # 'instanceID',
    'context_id',
    'context_page_id',
    # 'context_timestamp',
    #    'instance_id',
    # 'is_trade',
    'item_brand_id',
    # 'item_category_list',
       'item_city_id',
    'item_collected_level',
    'item_id',
    'item_price_level',
       # 'item_property_list',
    'item_pv_level',
    'item_sales_level',
       # 'predict_category_property',
    'shop_id',
    'shop_review_num_level',
       'shop_review_positive_rate',
    'shop_score_delivery',
       'shop_score_description', 'shop_score_service', 'shop_star_level',
       'user_age_level', 'user_gender_id', 'user_id', 'user_occupation_id',
       'user_star_level',
    # 'context_timestamp_string',
    # 'context_year',
    #    'context_month',
    'context_day',
    # 'context_hour',
    # 'context_min',
    #    'context_sec',
    # 'item_category_split',
    'category_0',
    'category_1',
       'category_2',
    # 'item_property_split',
    'item_property_num',
       # 'predict_category_property_split',
    'predict_category_property_num',
#     
'pre_usr_clk_cnt', 'pre_usr_clk_same_item_id_cnt', 'his_usr_clk_cnt',
       'his_usr_clk_same_item_id_cnt', 'pre_usr_act_cnt',
       'pre_usr_act_same_item_id_cnt', 'his_usr_clk_same_item_id_fir_las',
       'his_usr_clk_same_item_id_seq', 'item_id_clk_gap_bf',
       'item_id_clk_gap_af', 'item_id_act_gap_bf', 'item_id_clk_cnt_bf',
       'item_id_clk_cnt_af', 'item_id_clk_gap_2_fir', 'item_id_clk_cnt_bf_3h',
       'item_id_clk_cnt_af_3h', 'clk_cnt_bf', 'clk_cnt_af', 'clk_cnt_bf_3h',
       'clk_cnt_af_3h'
#
]

label_col = 'is_trade'

df_train = my_utils.select_range_by_day(df, 18, 23)
df_test = my_utils.select_range_by_day(df, 24, 24)
print(df_test.head())
X, y = get_data_and_label(df_train, feature_cols, label_col)

X_test, y_test = get_data_and_label(df_test, feature_cols, label_col)

pred = []

num_round = 15
for random_seed in range(1,num_round):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.9, random_state=random_seed)
    model = lgb_clf
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              early_stopping_rounds=50,
              verbose=50,
              )
    pred_ = model.predict_proba(X_test)[:, 1]
    pred_loss = logloss(y_test, pred_)
    print('random_seed:%d, log-loss:%f'%(random_seed, pred_loss))
    if len(pred) == 0:
        pred = logit(pred_)
    else:
        pred += logit(pred_)

# logloss_baseline(y_train, y_val)
pred = logistic(pred/num_round)
pred_loss = logloss(y_test, pred)
print(pred_loss)


'''
原始数据 valid_0's binary_logloss: 0.0824322
加入用户维度数据 valid_0's binary_logloss: 0.0801978
random_seed, 6times, 0.08020430123815359 
'''

# y_pred = np.round(y_pred, 12)
# y_pred = np.round(y_pred * my_utils.mean_cvr / np.mean(y_pred), 12)
# df = pd.DataFrame({"instanceID": df_instanceID["instanceID"].values, "proba": y_pred})
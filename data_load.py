#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd

df_train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", nrows=10000, sep=' ')

df_train.sort_values(by='instance_id', inplace=True)

def load_raw_data(df):

    # print df.head()

    print(df.columns)

    # 基础变量
    '''
    instance_id: 样本编号，Long
    # ID标志
    
    is_trade: 是否交易的标记位，Int类型；取值是0或者1，其中1 表示这条样本最终产生交易，0 表示没有交易
    # label，test_data没有
    
    item_id: 广告商品编号，Long类型
    # 关联商品信息
    
    user_id: 用户的编号，Long类型
    # 关联用户信息
    
    context_id: 上下文信息的编号，Long类型
    # 关联上下文信息
    
    shop_id: 店铺的编号，Long类型
    # 关联店铺信息
    '''
    cols_basic = ['instance_id', 'is_trade', 'item_id', 'user_id', 'context_id', 'shop_id']

    # 商品特征
    '''
    item_id: 广告商品编号，Long类型
    
    item_category_list: 
        广告商品的的类目列表，String类型；从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列，
        数据拼接格式为 "category_0;category_1;category_2"，
        其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目
    # 绝大多数只有category_0;category_1，极少部分数据有category_2
        
    item_property_list: 广告商品的属性列表，String类型；数据拼接格式为 "property_0;property_1;property_2"，
    各个属性没有从属关系
    # repeated类型，具体多少个不确定(约三四十个)，需要NLP处理
    
    item_brand_id: 广告商品的品牌编号，Long类型
    
    item_city_id: 广告商品的城市编号，Long类型
    
    item_price_level: 广告商品的价格等级，Int类型；取值从0开始，数值越大表示价格越高
    
    item_sales_level: 广告商品的销量等级，Int类型；取值从0开始，数值越大表示销量越大
    
    item_collected_level: 广告商品被收藏次数的等级，Int类型；取值从0开始，数值越大表示被收藏次数越大
    
    item_pv_level: 广告商品被展示次数的等级，Int类型；取值从0开始，数值越大表示被展示次数越大
    
    '''
    cols_ad_product = ['item_id', 'item_category_list', 'item_property_list', 'item_brand_id',
                       'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
                       'item_pv_level']

    # 用户特征
    '''
    user_id: 用户的编号，Long类型
    
    user_gender_id: 用户的预测性别编号，Int类型；0表示女性用户，1表示男性用户，2表示家庭用户
    
    user_age_level: 用户的预测年龄等级，Int类型；数值越大表示年龄越大
    
    user_occupation_id: 用户的预测职业编号，Int类型
    
    user_star_level: 用户的星级编号，Int类型；数值越大表示用户的星级越高
    
    '''
    cols_user = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']

    # 上下文特征
    # context_timestamp为时间单位，需
    '''
    context_id: 上下文信息的编号，Long类型
    
    context_timestamp: 广告商品的展示时间，Long类型；取值是以秒为单位的Unix时间戳，以1天为单位对时间戳进行了偏移
    # 时间戳，需做离散化处理，并转化为时间距离
    
    context_page_id: 广告商品的展示页面编号，Int类型；取值从1开始，依次增加；在一次搜索的展示结果中第一屏的编号为1，第二屏的编号为2
    # enum变量
    
    predict_category_property: 
        根据查询词预测的类目属性列表，
        String类型；数据拼接格式为 “category_A:property_A_1,property_A_2,property_A_3;category_B:-1;
                                  category_C:property_C_1,property_C_2” ，
        其中 category_A、category_B、category_C 是预测的三个类目；
        property_B 取值为-1，表示预测的第二个类目 category_B 没有对应的预测属性
    '''
    cols_context = ['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']

    # 店铺特征
    # 店铺特征都是int和long型，对gbdt模型可直接使用
    '''
    shop_id: 店铺的编号，Long类型
    
    shop_review_num_level: 店铺的评价数量等级，Int类型；取值从0开始，数值越大表示评价数量越多
    
    shop_review_positive_rate: 店铺的好评率，Double类型；取值在0到1之间，数值越大表示好评率越高
    
    shop_star_level: 店铺的星级编号，Int类型；取值从0开始，数值越大表示店铺的星级越高
    
    shop_score_service: 店铺的服务态度评分，Double类型；取值在0到1之间，数值越大表示评分越高
    
    shop_score_delivery: 店铺的物流服务评分，Double类型；取值在0到1之间，数值越大表示评分越高
    
    shop_score_description: 店铺的描述相符评分，Double类型；取值在0到1之间，数值越大表示评分越高
    
    '''
    cols_shop = ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

    print(df[cols_basic].head(1))
    print(df[cols_ad_product].head(1))
    print(df[cols_user].head(1))
    print(df[cols_context].head(1))
    print(df[cols_shop].head(1))


def pre_process_data(df):

    def split_str(x, s_str):
        return x.split(s_str)

    def get_arr_len(x):
        return len(x)

    def get_str_in_list(x, idx):
        if idx+1 > len(x):
            return '-1'
        return x[idx]

    # pre process category
    category_tree = df['item_category_list'].apply(split_str, args=(';',))
    print(category_tree.head())
    df['category_0'] = category_tree.apply(get_str_in_list, args=(0,))
    df['category_1'] = category_tree.apply(get_str_in_list, args=(1,))
    df['category_2'] = category_tree.apply(get_str_in_list, args=(2,))
    df.drop(['item_category_list'], axis=1, inplace=True)
    print(df[['category_0', 'category_1', 'category_2']].head())

    # pre process property
    df['item_property_list'] = df['item_property_list'].apply(split_str, args=(';',))
    print(df['item_property_list'].head())

    # pre process predict_category_property
    df['predict_category_property'] = df['predict_category_property'].apply(split_str, args=(';',))
    # print(df['predict_category_property'].apply(get_arr_len).mean())
    # 长度不定

    print(df.head())
    pass

# load_raw_data(df_train)
pre_process_data(df_train)



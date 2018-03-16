import pandas as pd
import datetime
import my_utils


def gen_usr_feature_intermediates_1var(df_, context_day):
    var = 'user_id'
    t_var1 = df_[[var, 'instanceID', 'context_timestamp', 'is_trade']]
    t_var1 = t_var1.groupby([var])

    t_var1_instanceID = t_var1['instanceID'].agg(lambda x: ':'.join(x)).reset_index()
    t_var1_instanceID.rename(columns={'instanceID': 'instanceIDs'}, inplace=True)
    save_dir = '../intermediates/t_var1_instanceID_%s.csv' % (context_day)
    t_var1_instanceID.to_csv(save_dir, index=False)
    del t_var1_instanceID
    print('%s saved.' % save_dir)

    t_var1_context_timestamp = t_var1['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    t_var1_context_timestamp.rename(columns={'context_timestamp': 'context_timestamps'}, inplace=True)
    save_dir = '../intermediates/t_var1_context_timestamp_%s.csv' % (context_day)
    t_var1_context_timestamp.to_csv(save_dir, index=False)
    del t_var1_context_timestamp
    print('%s saved.' % save_dir)

    t_var1_is_trade = t_var1['is_trade'].agg(lambda x: ':'.join(x)).reset_index()
    save_dir = '../intermediates/t_var1_is_trade_%s.csv' % (context_day)
    t_var1_is_trade.to_csv(save_dir, index=False)
    del t_var1_is_trade
    print('%s saved.' % save_dir)

    del t_var1


def gen_usr_feature_intermediates_2var(df_,  var_tmp, context_day):
    var = 'user_id'
    t_var2 = df_[[var, var_tmp, 'instanceID', 'context_timestamp', 'is_trade']]
    t_var2 = t_var2.groupby([var, var_tmp])

    t_var2_instanceID = t_var2['instanceID'].agg(lambda x: ':'.join(x)).reset_index()
    t_var2_instanceID.rename(columns={'instanceID': 'instanceIDs'}, inplace=True)
    save_dir = '../intermediates/t_var2_%s_instanceID_%s.csv' % (var_tmp, context_day)
    t_var2_instanceID.to_csv(save_dir, index=False)
    del t_var2_instanceID
    print('%s saved.' % save_dir)

    t_var2_context_timestamp = t_var2['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    t_var2_context_timestamp.rename(columns={'context_timestamp': 'context_timestamps'}, inplace=True)
    save_dir = '../intermediates/t_var2_%s_context_timestamp_%s.csv' % (var_tmp, context_day)
    t_var2_context_timestamp.to_csv(save_dir, index=False)
    del t_var2_context_timestamp
    print('%s saved.' % save_dir)

    t_var2_is_trade = t_var2['is_trade'].agg(lambda x: ':'.join(x)).reset_index()
    save_dir = '../intermediates/t_var2_%s_is_trade_%s.csv' % (var_tmp, context_day)
    t_var2_is_trade.to_csv(save_dir, index=False)
    del t_var2_is_trade
    print('%s saved.' % save_dir)

    del t_var2


def gen_usr_feature_click_var(df_, var_tmp, context_day):
    var = 'user_id'

    other_feature = df_.ix[df_['context_day'] == context_day, ['instanceID', var, var_tmp]]

    t_var = df_.ix[df_['context_day'].astype(int) < context_day, [var]]
    t_var['pre_usr_clk_cnt'] = 0
    t = t_var.groupby([var]).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=var, how='left')
    t_var = df_.ix[df_['context_day'] < context_day, [var, var_tmp]]
    t_var['pre_usr_clk_same_%s_cnt' % var_tmp] = 0
    t = t_var.groupby([var, var_tmp]).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=[var, var_tmp], how='left')

    t_var = df_[[var]]
    t_var['his_usr_clk_cnt'] = 0
    t = t_var.groupby([var]).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=var, how='left')
    df_ = pd.merge(df_, t, on=var, how='left')
    t_var = df_[[var, var_tmp]]
    t_var['his_usr_clk_same_%s_cnt' % var_tmp] = 0
    t = t_var.groupby([var, var_tmp]).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=[var, var_tmp], how='left')

    mask = (df_['context_day'] < context_day) & (df_['is_trade'] == '1')
    t_var = df_.ix[mask, [var]]
    t_var['pre_usr_act_cnt'] = 0
    t = t_var.groupby(var).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=[var], how='left')
    t_var = df_.ix[mask, [var, var_tmp]]
    t_var['pre_usr_act_same_%s_cnt' % var_tmp] = 0
    t = t_var.groupby([var, var_tmp]).count().reset_index()
    other_feature = pd.merge(other_feature, t, on=[var, var_tmp], how='left')

    del t_var, t

    other_feature.fillna(0, inplace=True)

    print('len :', len(df_))
    df_ = df_.ix[df_['his_usr_clk_cnt'] > 1, :]
    print('len of usr clk > 1 :', len(df_))

    def is_firstlastone(s):
        instanceID, instanceIDs = s.split('-')
        instanceIDs = instanceIDs.split(':')
        instanceID = int(instanceID)
        instanceIDs = [int(x) for x in instanceIDs]
        is_fir_last = 2
        clk_seq = 1
        for idx in instanceIDs:
            if instanceID == idx:
                break
            clk_seq += 1
        if instanceID == max(instanceIDs):
            is_fir_last = 3
        elif instanceID == min(instanceIDs):
            is_fir_last = 1
        return is_fir_last, clk_seq

    def get_time_gap_2_fir(s):
        context_timestamp, dates, instanceID, instanceIDs = s.split('_')
        context_timestamp = int(context_timestamp)
        instanceID = int(instanceID)
        dates = dates.split(':')
        dates = [int(x) for x in dates]
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        gaps = []
        for d, idx in zip(dates, instanceIDs):
            if instanceID > idx:
                this_gap = my_utils.calc_time_gap(context_timestamp, d)
                gaps.append(this_gap + 1)
        if len(gaps) == 0:
            return 0
        else:
            return max(gaps)

    def get_time_gap_bf(s):
        context_timestamp, dates, instanceID, instanceIDs = s.split('_')
        context_timestamp = int(context_timestamp)
        instanceID = int(instanceID)
        dates = dates.split(':')
        dates = [int(x) for x in dates]
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        gaps = []
        for d, idx in zip(dates, instanceIDs):
            if instanceID > idx:
                this_gap = my_utils.calc_time_gap(context_timestamp, d)
                gaps.append(this_gap + 1)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    def get_time_gap_af(s):
        context_timestamp, dates, instanceID, instanceIDs = s.split('_')
        context_timestamp = int(context_timestamp)
        instanceID = int(instanceID)
        dates = dates.split(':')
        dates = [int(x) for x in dates]
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        gaps = []
        for d, idx in zip(dates, instanceIDs):
            if instanceID < idx:
                this_gap = my_utils.calc_time_gap(d, context_timestamp)
                gaps.append(this_gap + 1)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    def get_act_gap_bf(s):
        context_timestamp, dates, is_trade = s.split('_')
        context_day = int(context_timestamp) // 1000000
        dates = dates.split(':')
        dates = [int(x) // 1000000 for x in dates]
        is_trade = is_trade.split(':')
        is_trade = [int(float(x)) for x in is_trade]
        gaps = []
        for d, _is_trade in zip(dates, is_trade):
            if (_is_trade == 1) & (d < context_day):
                gaps.append(context_day - d)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    def get_clk_cnt_bf(s):
        _, _, instanceID, instanceIDs = s.split('_')
        instanceID = int(instanceID)
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        count = 0
        for idx in instanceIDs:
            if instanceID > idx:
                count += 1
        if count == 0:
            return -1
        else:
            return count

    def get_clk_cnt_af(s):
        _, _, instanceID, instanceIDs = s.split('_')
        instanceID = int(instanceID)
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        count = 0
        for idx in instanceIDs:
            if instanceID < idx:
                count += 1
        if count == 0:
            return -1
        else:
            return count

    def get_clk_cnt_bf_sec(s, sec):
        context_timestamp, dates, instanceID, instanceIDs = s.split('_')
        context_timestamp = int(context_timestamp)
        instanceID = int(instanceID)
        dates = dates.split(':')
        dates = [int(x) for x in dates]
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        count = 0
        for d, idx in zip(dates, instanceIDs):
            if instanceID > idx:
                this_gap = my_utils.calc_time_gap(context_timestamp, d)
                if (this_gap >= 0) & (this_gap < sec):
                    count += 1
        if count == 0:
            return -1
        else:
            return count

    def get_clk_cnt_af_sec(s, sec):
        context_timestamp, dates, instanceID, instanceIDs = s.split('_')
        context_timestamp = int(context_timestamp)
        instanceID = int(instanceID)
        dates = dates.split(':')
        dates = [int(x) for x in dates]
        instanceIDs = instanceIDs.split(':')
        instanceIDs = [int(x) for x in instanceIDs]
        count = 0
        for d, idx in zip(dates, instanceIDs):
            if instanceID < idx:
                this_gap = my_utils.calc_time_gap(d, context_timestamp)
                if (this_gap >= 0) & (this_gap < sec):
                    count += 1
        if count == 0:
            return -1
        else:
            return count

    t_var2_instanceID = pd.read_csv(
        '../intermediates/t_var2_%s_instanceID_%s.csv' % (var_tmp, context_day))
    t_var2_context_timestamp = pd.read_csv(
        '../intermediates/t_var2_%s_context_timestamp_%s.csv' % (var_tmp, context_day))
    t_var2_is_trade = pd.read_csv(
        '../intermediates/t_var2_%s_is_trade_%s.csv' % (var_tmp, context_day))
    # t_var2_instanceID = my_utils.col2str(t_var2_instanceID)
    # t_var2_context_timestamp = my_utils.col2str(t_var2_context_timestamp)
    # t_var2_is_trade = my_utils.col2str(t_var2_is_trade)

    t3 = df_.ix[df_['context_day'] == context_day, [var, var_tmp, 'instanceID']]
    t3 = pd.merge(t3, t_var2_context_timestamp, on=[var, var_tmp], how='left')
    t3['instanceID_idx'] = (t3.instanceID.str.cat(t3.context_timestamps, sep='-'))
    t3['his_usr_clk_same_%s_fir_las' % var_tmp] = 0
    t3['his_usr_clk_same_%s_seq' % var_tmp] = 0
    t3['his_usr_clk_same_%s_fir_las' % var_tmp], t3['his_usr_clk_same_%s_seq' % var_tmp] = \
        zip(*t3.instanceID_idx.apply(is_firstlastone))
    t3 = t3[['instanceID',
             'his_usr_clk_same_%s_fir_las' % var_tmp,
             'his_usr_clk_same_%s_seq' % var_tmp]]
    other_feature = pd.merge(other_feature, t3, on=['instanceID'], how='left')
    del t3

    t7 = df_.ix[df_['context_day'] == context_day, [var, var_tmp, 'context_timestamp', 'instanceID']]
    t7 = pd.merge(t7, t_var2_context_timestamp, on=[var, var_tmp], how='left')
    t7 = pd.merge(t7, t_var2_instanceID, on=[var, var_tmp], how='left')
    t7 = pd.merge(t7, t_var2_is_trade, on=[var, var_tmp], how='left')
    t7['context_timestamp_date'] = (t7.context_timestamp.str.cat(t7.context_timestamps, sep='_'))
    t7['context_timestamp_date_is_trade'] = t7.context_timestamp_date.str.cat(t7.is_trade, sep='_')
    t7['context_timestamp_date'] = t7.context_timestamp_date.str.cat(t7.instanceID, sep='_')
    t7['context_timestamp_date'] = t7.context_timestamp_date.str.cat(t7.instanceIDs, sep='_')
    t7['%s_clk_gap_bf' % var_tmp] = t7.context_timestamp_date.apply(get_time_gap_bf)
    t7['%s_clk_gap_af' % var_tmp] = t7.context_timestamp_date.apply(get_time_gap_af)
    t7['%s_act_gap_bf' % var_tmp] = t7.context_timestamp_date_is_trade.apply(get_act_gap_bf)
    t7['%s_clk_gap_2_fir' % var_tmp] = t7.context_timestamp_date.apply(get_time_gap_2_fir)
    t7['%s_clk_cnt_bf' % var_tmp] = t7.context_timestamp_date.apply(get_clk_cnt_bf)
    t7['%s_clk_cnt_af' % var_tmp] = t7.context_timestamp_date.apply(get_clk_cnt_af)
    t7['%s_clk_cnt_bf_3h' % var_tmp] = t7.context_timestamp_date.apply(get_clk_cnt_bf_sec, sec=3 * 3600)
    t7['%s_clk_cnt_af_3h' % var_tmp] = t7.context_timestamp_date.apply(get_clk_cnt_af_sec, sec=3 * 3600)
    t7 = t7[['instanceID',
             '%s_clk_gap_bf' % var_tmp,
             '%s_clk_gap_af' % var_tmp,
             '%s_act_gap_bf' % var_tmp,
             '%s_clk_cnt_bf' % var_tmp,
             '%s_clk_cnt_af' % var_tmp,
             '%s_clk_gap_2_fir' % var_tmp,
             '%s_clk_cnt_bf_3h' % var_tmp,
             '%s_clk_cnt_af_3h' % var_tmp,
             ]]
    other_feature = pd.merge(other_feature, t7, on=['instanceID'], how='left')
    del t7
    del t_var2_is_trade, t_var2_context_timestamp, t_var2_instanceID


    t_var1_instanceID = pd.read_csv(
        '../intermediates/t_var1_instanceID_%s.csv' % (context_day))
    t_var1_context_timestamp = pd.read_csv(
        '../intermediates/t_var1_context_timestamp_%s.csv' % (context_day))
    t_var1_is_trade = pd.read_csv('../intermediates/t_var1_is_trade_%s.csv' % (context_day))

    t9 = df_.ix[df_['context_day'] == context_day, [var, 'context_timestamp', 'instanceID']]
    t9 = pd.merge(t9, t_var1_context_timestamp, on=[var], how='left')
    t9 = pd.merge(t9, t_var1_instanceID, on=[var], how='left')
    t9 = pd.merge(t9, t_var1_is_trade, on=[var], how='left')
    t9['context_timestamp_date'] = (t9.context_timestamp.str.cat(t9.context_timestamps, sep='_'))
    t9['context_timestamp_date_is_trade'] = t9.context_timestamp_date.str.cat(t9.is_trade, sep='_')
    t9['context_timestamp_date'] = t9.context_timestamp_date.str.cat(t9.instanceID, sep='_')
    t9['context_timestamp_date'] = t9.context_timestamp_date.str.cat(t9.instanceIDs, sep='_')
    t9['clk_cnt_bf'] = t9.context_timestamp_date.apply(get_clk_cnt_bf)
    t9['clk_cnt_af'] = t9.context_timestamp_date.apply(get_clk_cnt_af)
    t9['clk_cnt_bf_3h'] = t9.context_timestamp_date.apply(get_clk_cnt_bf_sec, sec=180 * 60)
    t9['clk_cnt_af_3h'] = t9.context_timestamp_date.apply(get_clk_cnt_af_sec, sec=180 * 60)
    # t9['clk_cnt_today'] = t9.context_timestamp_date.apply(get_clk_cnt_today)
    t9 = t9[['instanceID',
             'clk_cnt_bf',
             'clk_cnt_af',
             'clk_cnt_bf_3h',
             'clk_cnt_af_3h',
             # 'clk_cnt_today',
             ]]

    other_feature = pd.merge(other_feature, t9, on=['instanceID'], how='left')
    del t9
    del t_var1_is_trade, t_var1_context_timestamp, t_var1_instanceID

    other_feature = other_feature.drop([var,
                                        var_tmp,
                                        ], axis=1)

    return other_feature


def extract_feature_in_pre_days(df):
    for context_day in range(18,26):
        df_ = my_utils.select_range_by_day(df, context_day-3, context_day)
        df_ = my_utils.fix_is_trade(df_, context_day)
        str_cols = ['context_timestamp', 'instanceID', 'is_trade', 'user_id', 'item_id',
            'user_id', 'context_id', 'shop_id', 'item_brand_id',
            'category_0', 'category_1', 'category_2']
        # for col in str_cols:
        #     if df_[col].dtype != str:
        #         df_.ix[:, col] = df_[col].values.astype(str)
        df_.context_timestamp = df_.context_timestamp.astype(str)
        df_.instanceID = df_.instanceID.astype(str)
        df_.is_trade = df_.is_trade.astype(str)
        gen_usr_feature_intermediates_1var(df_, context_day)
        gen_usr_feature_intermediates_2var(df_, 'item_id', context_day)
        gen_usr_feature_intermediates_2var(df_, 'item_brand_id', context_day)
        gen_usr_feature_intermediates_2var(df_, 'shop_id', context_day)
        df_interest = gen_usr_feature_click_var(df_, 'item_id', context_day)
        df_interest.fillna(0, inplace=True)
        save_dir = '../intermediates/usr_feature_%d.csv' % (context_day)
        df_interest.to_csv(save_dir, index=False, chunksize=10 ** 6)
        print('-' * 3, save_dir, '-' * 3)
    pass


use_cols = ['context_timestamp', 'instanceID', 'context_day', 'is_trade', 'user_id', 'item_id',
            'user_id', 'context_id', 'shop_id', 'item_brand_id',
            'category_0', 'category_1', 'category_2']
df = pd.read_csv("../data/df_concat.csv", usecols=use_cols)
print(df['context_day'].unique())
extract_feature_in_pre_days(df)

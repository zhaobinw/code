import pandas as pd
import numpy as np
import bayes_smoothing as bs

print('LOADING')
df = pd.read_csv("../data/df_concat.csv")
df_merge = pd.DataFrame()
for context_day in range(18,26):
    df_ = df.ix[df['context_day'] == context_day]
    df_interest = pd.read_csv( '../intermediates/usr_feature_%s.csv' % (context_day))
    df_ = df_.merge(df_interest, on=['instanceID'], how='left')
    del df_interest
    # df_['pre_usr_cvr'] = bs.smooth(df_['pre_usr_act_cnt'].values,
    #                               df_['pre_usr_clk_cnt'].values,
    #                               0.0091,
    #                               0.3288
    #                               )
    print(context_day, 'DONE')
    if len(df_merge)==0:
        df_merge = df_
    else:
        df_merge = pd.concat([df_merge, df_], axis=0)

print(df_merge.head())
print(df_merge.columns)
df_merge.to_csv('../data/df_merge.csv', index=False)
print('COMPLETE')
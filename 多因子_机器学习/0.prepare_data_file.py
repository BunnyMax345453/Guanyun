import os
os.chdir('D:/Python/Flies/Guanyun/Barra_Multi_Factor')
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

default_df = pd.read_csv('dataset/factor_data/descrp_factor/book_leverage.csv')
stocks = default_df.columns
# date_begin = default_df.date.to_list()[0]  # 2011-01-04
date_begin = '2011-01-01'
# date_begin = '2021-01-01'
date_end = '2021-06-20'

################
def prepare_features(date_begin, date_end, stocks, factor = 'barra') : 
    """
    function: 调整barra因子dataframe格式并合并到一张dataframe上
    input: 开始和结束日期，以及股票
    output: 输出barra因子dataframe
            columns: date,  stockname,  barra factors ...
            index: 0,1,2...
    """
    if factor == 'barra':
        route = 'dataset/factor_data/barra_factor/'
    elif factor == 'descrp':
        route = 'dataset/factor_data/descrp_factor/'
    elif factor == 'alpha101':
        route = 'dataset/factor_data/alpha101/'

    for root, dirs, files in os.walk(route):
        pass
    for i_num in range(len(files)):
        # 对每个factor循环，读取csv文件
        i = files[i_num]
        print(i)
        key = i[:-4]
        factor_df = pd.read_csv(route + i)
        # 截取对应的日期段和股票集合
        factor_df = factor_df[(factor_df['date'] >= date_begin) & (factor_df['date'] <= date_end)]
        factor_df = factor_df[stocks]
        # 对factor每日进行标准化处理
        factor_df.iloc[:,1:] = factor_df.iloc[:,1:]\
                                .sub(factor_df.iloc[:,1:].mean(axis=1),axis=0)\
                                .div(factor_df.iloc[:,1:].std(axis=1,ddof=1), axis=0)
        # 将factor_df转化为date, stockname, factor_1的一维数据格式
        df_tmp = pd.melt(factor_df, 
                        id_vars=['date'], 
                        value_vars=list(factor_df.columns[1:]),
                        var_name='stockname', 
                        value_name=key) 
        df_tmp = df_tmp.set_index(df_tmp['date'] + df_tmp['stockname'])
        # 去除重复数据
        df_tmp = df_tmp.groupby(df_tmp.index).first()
        if i_num == 0 :
            sum_df = pd.DataFrame()
            sum_df[['date','stockname']] = df_tmp[['date','stockname']]
            sum_df = sum_df.set_index(sum_df['date'] + sum_df['stockname'])
        # 将各个factor_df拼接为一个dataframe
        sum_df = pd.merge(sum_df, df_tmp.iloc[:,2:], how='left', left_index=True, right_index=True)
        

    sum_df.sort_values(by=['stockname', 'date'],ascending=True)
    sum_df.iloc[:,2:] = sum_df.iloc[:,2:].groupby(sum_df['stockname']).transform(lambda x:x.fillna(method='ffill')) # fillna
    sum_df = sum_df.set_index(sum_df['date'] + sum_df['stockname'])    
    return sum_df

# prepare_features(date_begin, date_end, stocks, factor = 'alpha101')
###################
def prepare_targets(date_begin, date_end, stocks, rolling_list=[1,5,22], D=[0]) : 
    """
    function: 调整因变量dataframe格式
    input: 开始和结束日期，以及股票;rolling的天数(list)，滞后时间(至少应为1天)
            如果rolling=5,D=1,则2020-01-01的数据为2020-01-02/03/04/05/06共5天的收益总和(假设全为交易日)
    output: 输出因变量dataframe
            columns: date,  stockname, return_rollingX_delayY(rollingX天delayY天的股票收益), 
                relative_target_rollingX_delayY(rollingX天delayY天的股票相对指数收益哑变量), 
                relative_return_rollingX_delayY(rollingX天delayY天的股票相对指数收益)
            index: 0,1,2...
    """
    # 读取个股收盘价并转为return，节选日期段和股票
    return_df = pd.read_csv('D:/Python/Flies/Guanyun/Barra_Multi_Factor/dataset/factor_data/ml_close_data/stock_close_data.csv')\
                .sort_values(by='date')
    return_df.iloc[:,1:] = return_df.iloc[:,1:].pct_change()
    return_df = return_df[(return_df['date'] >= date_begin) & (return_df['date'] <= date_end)][stocks]
    
    # 读取指数收盘价并转为return，节选日期段
    index_return_df = pd.read_csv('dataset/factor_data/ml_close_data/' + 'index_close_data.csv')[['date','close']] # columns:date,open,close,high,low,volume,money, index:0,1,2...
    index_return_df['return'] = index_return_df['close'].pct_change()
    index_return_df = index_return_df[['date','return']]
    index_return_df = index_return_df[(index_return_df['date'] >= date_begin) & (index_return_df['date'] <= date_end)]
    
    # 相对涨幅：指数的成分股涨幅相对于指数涨幅
    relative_return_df = pd.merge(index_return_df[['date']], return_df.iloc[:,1:].sub(index_return_df['return'],axis=0),
                                 left_index=True, right_index=True)  # 相对涨幅

    res = pd.DataFrame()
    for rolling_num in rolling_list:
        for D_num in D:
            # 对rolling_list=[1,5,22], D=[1]的变量分别生成不同rollling和delay的输出变量，
            # 每种分别有return，和指数的相对收益relative_return，以及对应的哑变量relative_target
            
            # 给各个因变量命名
            columns_name = 'return_rolling'+str(rolling_num)+'_delay'+str(D_num)
            columns_name_target = 'relative_target_rolling'+str(rolling_num)+'_delay'+str(D_num)
            columns_name_rel_return = 'relative_return_rolling'+str(rolling_num)+'_delay'+str(D_num)

            return_df_tmp = return_df.copy(deep=True)
            return_df_tmp.iloc[:,1:] = return_df_tmp.iloc[:,1:].rolling(rolling_num).sum()
            return_df_tmp.iloc[:,1:] = return_df_tmp.iloc[:,1:].shift(-(rolling_num+D_num))

            res_return = pd.melt(return_df_tmp, 
                        id_vars=['date'], 
                        value_vars=list(return_df_tmp.columns[1:]), # list of days of the week
                        var_name='stockname', 
                        value_name=columns_name)
            if len(res) == 0 : 
                res = res_return
            else :
                res = pd.merge(res, res_return, how='left', on=['date','stockname'])
            
            index_return_df_tmp = index_return_df.copy(deep=True)
            index_return_df_tmp.iloc[:,1:] = index_return_df_tmp.iloc[:,1:].rolling(rolling_num).sum()
            index_return_df_tmp.iloc[:,1:] = index_return_df_tmp.iloc[:,1:].shift(-(rolling_num+D_num))
            
            relative_return_df = pd.merge(
                            return_df_tmp.iloc[:,0],return_df_tmp.iloc[:,1:].sub(index_return_df_tmp['return'],axis=0),
                            left_index=True, right_index=True)
            res_relativereturn = pd.melt(relative_return_df, 
                        id_vars=['date'], 
                        value_vars=list(relative_return_df.columns[1:]), # list of days of the week
                        var_name='stockname', 
                        value_name=columns_name_rel_return)
            # 若差值<0则为0，反之为1
            res_relativereturn[columns_name_target] = 0
            res_relativereturn.loc[res_relativereturn[columns_name_rel_return]>0, columns_name_target] = 1
            res = pd.merge(res, res_relativereturn, how='left', on=['date','stockname'])

    res.sort_values(by=['stockname', 'date'],ascending=True)
    res = res.set_index(res['date'] + res['stockname'])
    return res
# prepare_targets(date_begin, date_end, stocks)
################################
def alpha101_reg_epsilon(date_begin = date_begin, date_end = date_end, stocks = stocks, alpha_plus_barra=True):
    """
    function: alpha101因子对barra的描述因子回归并取残差作为新的alpha101因子，将剔除barra影响后的因子与描述因子拼接组成新的feature
          若alpha_plus_barra为False，则只输出alpha101的残差
    input: 开始和结束日期，以及股票
    output: 输出自变量dataframe，因变量dataframe
    """
    barra_df = prepare_features(date_begin, date_end, stocks, factor = 'barra')
    alpha101_df = prepare_features(date_begin, date_end, stocks, factor = 'alpha101')
    
    barra_df = barra_df.dropna(how = 'any',axis = 0)
    alpha101_df = alpha101_df.dropna(how = 'any',axis = 0)
    index_list = list(set(barra_df.index) & set(alpha101_df.index))
    
    alpha101_df = alpha101_df.loc[index_list,:].sort_values(by=['stockname','date'])
    barra_df = barra_df.loc[index_list,:].sort_values(by=['stockname','date'])
    
    X = barra_df.iloc[:,2:]
    for alpha_factor in alpha101_df.columns[2:] :
        y = alpha101_df[alpha_factor]
        model = LinearRegression(fit_intercept=False)
        model.fit(X, alpha101_df[alpha_factor])

        factor_ret = model.coef_   

        epsilon = model.predict(X) - y
        standard_epsilon = (epsilon - epsilon.mean()) / epsilon.std()

        y = standard_epsilon
        alpha101_df[alpha_factor] = y
        
    if alpha_plus_barra == True :
        res = pd.merge(alpha101_df,barra_df,how='left', on=['date','stockname'])
    else:
        res = alpha101_df
    res = res.set_index(res['date'] + res['stockname'])

    def standard_func(df):
        df.iloc[:,2:] = df.iloc[:,2:].sub(df.iloc[:,2:].mean(axis=0),axis=1).div(df.iloc[:,2:].std(axis=0,ddof=1), axis=1)
        return df
    res = res.groupby('date').apply(standard_func)
    return res
# aa = alpha101_reg_epsilon(date_begin = date_begin, date_end = date_end, stocks = stocks)
#######################
#######################
def prepare_data_func(date_begin = date_begin, date_end = date_end, stocks = stocks, factor = 'descrp', alpha_plus_barra=True) : 
    """
    function: 在调整自变量因变量格式后，剔除空值行，排序整理
    input: 开始和结束日期，以及股票
    output: 输出自变量dataframe，因变量dataframe
    """
    if factor == 'alpha101':
        features_df = alpha101_reg_epsilon(date_begin = date_begin, date_end = date_end, stocks = stocks, alpha_plus_barra=True)
    else : 
        features_df = prepare_features(date_begin, date_end, stocks, factor = factor)
    targets_df = prepare_targets(date_begin, date_end, stocks)


    features_df = features_df.dropna(how = 'any',axis = 0)
    targets_df = targets_df.dropna(how = 'any',axis = 0)
    index_list = list(set(features_df.index) & set(targets_df.index))
    targets_df = targets_df.loc[index_list,:].sort_values(by=['stockname','date'])
    features_df = features_df.loc[index_list,:].sort_values(by=['stockname','date'])
    return features_df, targets_df

# prepare_data_func(date_begin = date_begin, date_end = date_end, stocks = stocks, factor = 'descrp')
import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot
import alphalens
os.chdir('D:/Python/Flies/Guanyun/Barra_Multi_Factor/')
from alphalens import performance
from alphalens import tears
from alphalens import plotting
from alphalens import utils

def factor_analysis_func(factor_data) : 
    '''
    func: 通过alphalen进行因子分析(现该函数已基本弃用，相关功能在被接下来自己写的函数功能取代)
        生成factor因子分析
    factor_data: dataframe 
    '''
    pricing = pd.read_csv('dataset/factor_data/ml_close_data/stock_close_data.csv').set_index('date')
    
    pricing.index = pd.to_datetime(pricing.index,format='%Y-%m-%d')
    factor_data.index = pd.to_datetime(factor_data.index,format='%Y-%m-%d')
    
    sec_index = list(set(factor_data.index).intersection(set(pricing.index)))  
    factor_data = factor_data.loc[sec_index].sort_index()
    pricing = pricing.loc[sec_index].sort_index()
    
    factor_data.index.set_names(['date'], inplace=True)
    factor_data = factor_data.stack()
    factor_data = pd.DataFrame(factor_data)
    # 设置索引名称
    factor_data.index.set_names(['date', 'asset'], inplace=True)
    # 设置列名称
    factor_data.columns = ['factor_value']
    print('factor analysis:')
    # Ingest and format data
    factor_return = alphalens.utils.get_clean_factor_and_forward_returns(factor_data,
                                                   pricing,
                                                   quantiles=5,
#                                                                    groupby=ticker_sector,
#                                                                    groupby_labels=sector_names
                                                      )
    print('###########IC###########')
    IC=performance.factor_information_coefficient(factor_return)
    print(IC.head(5))
    plotting.plot_ic_ts(IC)

    plotting.plot_ic_hist(IC)
    a=IC.iloc[:,0]
    print(len(a[a>0.02])/len(a))
    alphalens.tears.create_returns_tear_sheet(factor_return)
    
def data_processing(factor_data, D=1):
    """
    func: 读取收盘价信息，处理滞后D天并和因子dataframe的index对齐
    input: factor_data: 因子dataframe
        D: 收益dataframe的shift滞后天数，D=1表示因子预测一天后收益情况
    output: 对齐排序处理过后的因子df和收益df
    """
    # 读取收盘价数据，排序，计算每日涨跌幅，计算今天加未来共rolling天总收益，
    pricing = pd.read_csv('D:/Python/Flies/Guanyun/Barra_Multi_Factor/dataset/factor_data/ml_close_data/stock_close_data.csv')\
                .sort_values(by='date')  # value: daily close price; index: 0,1,2,..; columns: stocks
    
    pricing['date'] = pd.to_datetime(pricing['date'],format='%Y-%m-%d')  
    pricing = pricing.set_index('date')
    return_df = pricing.pct_change()
    return_df = return_df.shift(-D)
    
    factor_data.index = pd.to_datetime(factor_data.index,format='%Y-%m-%d')
    
    sec_index = list(set(factor_data.index)&(set(return_df.index)))  
    sec_columns = list(set(factor_data.columns)&(set(return_df.columns)))  
    factor_data = factor_data.loc[sec_index,sec_columns].sort_index().sort_index(axis=1)
    return_df = return_df.loc[sec_index,sec_columns].sort_index().sort_index(axis=1)
    
#     factor_data['date'] = factor_data.index
#     return_df['date'] = return_df.index
    return return_df, factor_data


def ic_calculate(factor_data,  D=1):
    """
    func: 计算信息因子IC
    input: factor_data: 因子dataframe
        D: 收益dataframe的shift滞后天数，D=1表示因子预测一天后收益情况
    output: 信息因子IC的时间序列
    """
    pricing, factor_data = data_processing(factor_data,D=D)
    # 将factor和return合为一个df: df_sum
    factor_data['date'] = factor_data.index
    pricing['date'] = pricing.index
    df_sum = pd.melt(factor_data, 
                    id_vars=['date'], 
                    value_vars=list(factor_data.columns[:-1]),
                    var_name='stockname', 
                    value_name='factor') 
    tmp = pd.melt(pricing, 
                id_vars=['date'], 
                value_vars=list(pricing.columns[:-1]),
                var_name='stockname', 
                value_name='pricing') 
    df_sum = pd.merge(df_sum, tmp, how='left', left_on=['date','stockname'], right_on=['date','stockname'])
    # columns: [date, stock, factor, return], index: 0,1,2...
    
    
    # 对每天计算factor和D天后收益的相关系数
    def df_corr_func(df):#函数的参数是一个DataFrame
        df = df.dropna(how='any',axis=0)
        res = np.corrcoef(df['factor'],df['pricing'])[0,1]
        return res
    
    corr_series = df_sum.groupby(df_sum['date']).apply(df_corr_func)
    return corr_series


def factor_data_processing(factor_data, rolling = 5):
    """
    将factor_data整理为几天一变动的dataframe，删除多余数据，从而实现几天一调仓的操作
    """
    aa = np.array([i for i in range(len(factor_data))])
    aa = aa[aa%rolling!=0]
    factor_data.iloc[aa] = np.nan    
    factor_data.fillna(method='ffill', inplace=True) # fillna
    return factor_data

def cumulative_return_calculate(factor_data, quantiles=5, D=1, fee = 0.003):
    """
    func: 生成因子的累计收益
    input: factor_data: 因子dataframe，columns为stock，index为date，value为因子值
        quantiles: 将每天的因子按高低分成的份数
        D: 滞后天数(这里按目前的做法应当一直取0)
        fee: 手续费，0时表示不考虑，否则为具体值
    output: 累计收益dataframe，index为date，columns为quantiles对应份数，value为累计收益
    """
    # 获取因子和收益df
    return_df, factor_data = data_processing(factor_data, D=D)
    factor_data = factor_data_processing(factor_data)
    
    # 将factor和return合为一个df: df_sum
    factor_data['date'] = factor_data.index
    return_df['date'] = return_df.index
    df_sum = pd.melt(factor_data, 
                    id_vars=['date'], 
                    value_vars=list(factor_data.columns[:-1]),
                    var_name='stockname', 
                    value_name='factor') 
    tmp = pd.melt(return_df, 
                id_vars=['date'], 
                value_vars=list(return_df.columns[:-1]),
                var_name='stockname', 
                value_name='return') 
    df_sum = pd.merge(df_sum, tmp, how='left', left_on=['date','stockname'], right_on=['date','stockname'])

    # 对stock按因子值大小分为quantiles类，计算每类的收益均值作为做多该类的收益，生成df: cumu_sum_df(columns: [1,2,...quantiles], index: date)
    def port_return_func(df, quantiles=quantiles) : 
        df['labels'] = pd.qcut(df['factor'],quantiles,labels=[i for i in range(1, quantiles+1)])
        group = df.groupby('labels')
        return group.mean()['return']
    cumu_sum_df = df_sum.groupby(df_sum['date']).apply(port_return_func)
    
    # 对各个quantiles里的股票生成set，通过与前一日相减获得变动的stock数量，除以目前持有stock数以计算换手率turnover_rate_df
    def port_content_func(df, quantiles=quantiles) : 
        df['labels'] = pd.qcut(df['factor'],quantiles,labels=[i for i in range(1, quantiles+1)])
        group = df.groupby('labels')
        res = pd.DataFrame(index = [0],columns=[i for i in range(1, quantiles+1)])
        for group_ in group:
            res.loc[0,group_[0]] = set(group_[1]['stockname'])
        return res
    port_content_df = df_sum.groupby(df_sum['date']).apply(port_content_func)
    port_content_df.index = [col[0] for col in port_content_df.index.values] # 产生的Multi Index改正为一维Index
    turnover_rate_df = port_content_df.diff().iloc[1:,:].applymap(lambda x:len(x)) / port_content_df.iloc[1:,:].applymap(lambda x:len(x))

    # 对于每个分类计算扣除手续费之后的收益；以及计算最大分类减最小分类，减去两个分类的手续费，之后所获得的收益值
    cumu_sum_df = cumu_sum_df.iloc[1:,:]
    cumu_5_1 = cumu_sum_df[quantiles] - turnover_rate_df[quantiles]*fee - cumu_sum_df[1] - turnover_rate_df[1]*fee
    cumu_sum_df = cumu_sum_df - turnover_rate_df*fee
    print('turnover rate:',turnover_rate_df)
    
    # 累计收益，作图
    cumu_5_1.cumsum().plot()
    cumu_sum_df.cumsum().plot()
    plt.show()
    return cumu_sum_df

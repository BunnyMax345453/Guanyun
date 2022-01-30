#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
从joinquant上下载的财报数据和财报因子数据格式不一致且不是我们所需要的，
这里进行格式调整，通过pivot和其他方法将格式整理并统一，对每个因子单独生成一个csv文件并保存
"""

import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot as plt
from pandas.core.groupby.groupby import DataError


# In[ ]:





# In[2]:


## 来自JQDATA的数据处理：
# 来自JQDATA的三表数据：
balance_dict = {
    'total_owner_equities': pd.DataFrame(),
    'total_assets': pd.DataFrame(),
    'equities_parent_company_owners': pd.DataFrame(),
    'total_liability': pd.DataFrame(),
    'inventories': pd.DataFrame(),
    'total_non_current_liability': pd.DataFrame(),
    'fixed_assets': pd.DataFrame(),
    'construction_materials': pd.DataFrame(),
    'constru_in_process': pd.DataFrame(),
    'total_assets': pd.DataFrame(),
    'total_current_assets': pd.DataFrame(),
    'total_current_liability': pd.DataFrame(),
    'preferred_shares_equity': pd.DataFrame()
}
income_dict = {
    'operating_revenue': pd.DataFrame(),
    'operating_cost': pd.DataFrame(),
    'net_profit': pd.DataFrame(),
    'total_operating_revenue': pd.DataFrame(),
}
cf_dict = {
    'inventory_decrease': pd.DataFrame(),
    'net_operate_cash_flow': pd.DataFrame(),
}

for root, dirs, files in os.walk('dataset/A股各公司财务数据/合并资产负债表/'):
    for f in files:
        data_df = pd.read_csv('dataset/A股各公司财务数据/合并资产负债表/' + f, encoding='gbk')
        for key in balance_dict.keys():
            try:
                pt = pd.pivot_table(data_df, index=['pub_date'], columns=['code'], values=[key], margins=False)
                balance_dict[key] = pd.concat([balance_dict[key], pt],axis=1)
            except DataError: 
                print(f, key, 'No data')
                
for root, dirs, files in os.walk('dataset/A股各公司财务数据/合并现金流量表/'):
    for f in files:
        data_df = pd.read_csv('dataset/A股各公司财务数据/合并现金流量表/' + f, encoding='gbk')
        for key in cf_dict.keys():
            try:
                pt = pd.pivot_table(data_df, index=['pub_date'], columns=['code'], values=[key], margins=False)
                cf_dict[key] = pd.concat([cf_dict[key], pt],axis=1)
            except DataError: 
                print(f, key, 'No data')
                
for root, dirs, files in os.walk('dataset/A股各公司财务数据/合并利润表/'):
    for f in files:
        data_df = pd.read_csv('dataset/A股各公司财务数据/合并利润表/' + f, encoding='gbk')
        for key in income_dict.keys():
            try:
                pt = pd.pivot_table(data_df, index=['pub_date'], columns=['code'], values=[key], margins=False)
                income_dict[key] = pd.concat([income_dict[key], pt],axis=1)
            except DataError: 
                print(f, key, 'No data')


# In[ ]:


# 来自JQDATA的财务指标数据：
def financial_indicator(ind) : 
    stock_df = pd.DataFrame()
    for root, dirs, files in os.walk('dataset/A股各公司财务数据/财务指标数据/'):
        for f in files:
            data_df = pd.read_csv('dataset/A股各公司财务数据/财务指标数据/' + f, encoding='gbk')
            pt = pd.pivot_table(data_df, index=['pubDate'], columns=['code'], values=[ind], margins=False)
            stock_df = pd.concat([stock_df, pt]) # join
    return stock_df.sort_index()

indicator_keys = ['roe_df',
    'roa_df',
    'net_profit_margin_df',
    'gross_profit_margin_df',
    'inc_net_profit_year_on_year_df',
    'financing_expense_to_total_revenue_df',
    'operating_expense_to_total_revenue_df',
    'eps']

indicator_dict = {}
for key in indicator_keys : 
    indicator_dict[key] = financial_indicator(key)


# In[ ]:


# 来自JQDATA的估值数据（只用到了pcf_ratio）：
def financial_valuation(ind) : 
    stock_df = pd.DataFrame()
    for root, dirs, files in os.walk('dataset/A股各公司财务数据/估值数据/'):
        for f in files:
#             print(f)
            data_df = pd.read_csv('dataset/A股各公司财务数据/估值数据/' + f, encoding='gbk')
            pt = pd.pivot_table(data_df, index=['day'], columns=['code'], values=[ind], margins=False)
            stock_df = pd.concat([stock_df, pt]) # join
    return stock_df.sort_index()

valuation_keys = ['pcf_ratio']
for key in indicator_keys : 
    indicator_dict[key] = financial_valuation(key)


# In[ ]:





# In[ ]:


# index上和交易日历的时间对齐，并ffill填充空白值

td = pd.read_csv('dataset/trading_date.csv', low_memory=False)    
td['date'] = pd.to_datetime(td['date'],format='%Y-%m-%d')

for key in balance_dict.keys() : 
    balance_dict[key].index = pd.to_datetime(balance_dict[key].index,format='%Y-%m-%d')
    new_df = pd.DataFrame(index=td.iloc[252:,0])
    balance_dict[key] = pd.merge(new_df,balance_dict[key], how='left', left_index=True, right_index=True).sort_index() # 合并交易日历和数据
    balance_dict[key].fillna(method='pad', inplace=True)   # ffill
    balance_dict[key].to_csv('dataset/data of factor need/'+key+'.csv')
    
for key in cf_dict.keys() : 
    cf_dict[key].index = pd.to_datetime(cf_dict[key].index,format='%Y-%m-%d')
    new_df = pd.DataFrame(index=td.iloc[252:,0])
    cf_dict[key] = pd.merge(new_df,cf_dict[key], how='left', left_index=True, right_index=True).sort_index() # 合并交易日历和数据
    cf_dict[key].fillna(method='pad', inplace=True)   # ffill
    cf_dict[key].to_csv('dataset/data of factor need/'+key+'.csv')
    
for key in income_dict.keys() : 
    income_dict[key].index = pd.to_datetime(income_dict[key].index,format='%Y-%m-%d')
    new_df = pd.DataFrame(index=td.iloc[252:,0])
    income_dict[key] = pd.merge(new_df,income_dict[key], how='left', left_index=True, right_index=True).sort_index() # 合并交易日历和数据
    income_dict[key].fillna(method='pad', inplace=True)   # ffill
    income_dict[key].to_csv('dataset/data of factor need/'+key+'.csv')
    
for key in indicator_dict.keys() : 
    indicator_dict[key].index = pd.to_datetime(indicator_dict[key].index,format='%Y-%m-%d')
    new_df = pd.DataFrame(index=td.iloc[252:,0])
    indicator_dict[key] = pd.merge(new_df,indicator_dict[key], how='left', left_index=True, right_index=True).sort_index() # 合并交易日历和数据
    indicator_dict[key].fillna(method='pad', inplace=True)   # ffill
    indicator_dict[key].to_csv('dataset/data of factor need/'+key+'.csv')


# In[ ]:





# In[ ]:


# 来自wind的数据，已经下载完毕，需要在columns上与来自jqdata的数据保持股票上的对齐，且后缀从诸如000001.SZ改为000001.XSHE，index上和tradingdate对齐

wind_keys = ['A股流通股本(股)',
    'A股流通市值(元)',
    '成交金额(元)',
    '总股本(股)',
    '市盈率',
    '市净率',
    '成交量(股)',
    '涨跌幅(%)',
    '换手率(%)',
    '收盘价(元)']


# In[ ]:


wind_dict = {}
for key in wind_keys : 
    wind_dict[key] = pd.read_csv('dataset/后复权数据-分类/' + key + '.csv', low_memory=False).set_index('date')
    
    # index上和tradingdate对齐
    wind_dict[key].index = pd.to_datetime(wind_dict[key].index,format='%Y-%m-%d')
    new_df = pd.DataFrame(index=td.iloc[252:,0])
    wind_dict[key] = pd.merge(new_df,wind_dict[key], how='left', left_index=True, right_index=True).sort_index() # 合并交易日历和数据
    
    # 填充空白值
    wind_dict[key].fillna(method='pad', inplace=True)   # ffill
    
    # 后缀从诸如000001.SZ改为000001.XSHE
    column_list = wind_dict[key].columns.to_list()
    for i in range(len(column_list)) : 
        if column_list[i][0:1] == '0' or column_list[i][0:1] == '3' : 
            column_list[i] = column_list[i][0:6] + '.XSHE' 
        elif column_list[i][0:1] == '6' : 
            column_list[i] = column_list[i][0:6] + '.XSHG' 
    wind_dict[key].columns = column_list
    
    wind_dict[key].to_csv('dataset/data of factor need/'+key+'.csv')


# In[ ]:


# 由于来自wind的股票数据和来自jqdata的股票数据，股票数量不是一模一样的，以及下载数据的这一段时间里存在新上市的股票等因素，这里将两者的股票取交集，对齐columns

for key in ['涨跌幅(%)','total_assets'] : 
    balance_dict[key] = pd.read_csv('dataset/data of factor need/' + key + '.csv').set_index('date').astype('float64')
    
stks = list(set(balance_dict['涨跌幅(%)'].columns).intersection(set(balance_dict['total_assets'].columns)))
for key in balance_dict.keys() : 
    if key != 'R_f' and key != 'R_m' : 
        try :
            balance_dict[key][stks]
        except : 
                other_list = list(set(stks).difference(set(balance_dict[key].columns)))
                print(key, len(other_list))
                balance_dict[key][other_list] = np.nan
        balance_dict[key].to_csv('dataset/data of factor need/' + key + '.csv')


# In[ ]:


"""
一致预测一年期FY1, 三年期FY3, 未来12个月FY12是从wind上每个月手动抓下来的，也是经过
保持股票上的对齐，且后缀从诸如000001.SZ改为000001.XSHE，index上和tradingdate对齐，三个步骤。
这里是直接在原数据上进行处理了，而且手动抓取的数据格式和自动下载的不太一样，所以没有另外列出来了。
"""


# In[ ]:





# In[ ]:





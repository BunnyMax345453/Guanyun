#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
从joinquant下载所需数据的code
"""

import pandas as pd
import numpy as np
import datetime
import math
import math
import os
from jqdatasdk import finance


# In[2]:


## 登陆jqdata
from jqdatasdk import *
auth('15071473211','F47isswjj') #ID是申请时所填写的手机号；Password为聚宽官网登录密码，新申请用户默认为手机号后6位


# In[3]:


## 获取所有股票代码
stock_data = get_all_securities(types=['stock'], date=None)
stock_list = stock_data.index.to_list()


# In[ ]:


## 获取财报三表


# In[ ]:


# 资产负债表
for stock in stock_list:
    q = query(finance.STK_BALANCE_SHEET).filter(
        finance.STK_BALANCE_SHEET.code == stock,
        finance.STK_BALANCE_SHEET.pub_date >= '2010-01-01',
        finance.STK_BALANCE_SHEET.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/合并资产负债表/' + stock + '.csv', encoding='gbk')


# In[8]:


# 金融类资产负债表
for stock in stock_list:
    q = query(finance.FINANCE_BALANCE_SHEET).filter(
        finance.FINANCE_BALANCE_SHEET.code == stock,
        finance.FINANCE_BALANCE_SHEET.pub_date >= '2010-01-01',
        finance.FINANCE_BALANCE_SHEET.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/金融类合并资产负债表/' + stock + '.csv', encoding='gbk')


# In[5]:


# 利润表
for stock in stock_list:
    q = query(finance.STK_INCOME_STATEMENT).filter(
        finance.STK_INCOME_STATEMENT.code == stock,
        finance.STK_INCOME_STATEMENT.pub_date >= '2010-01-01',
        finance.STK_INCOME_STATEMENT.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/合并利润表/' + stock + '.csv', encoding='gbk')


# In[13]:


# 金融类利润表
for root, dirs, files in os.walk('dataset/A股各公司财务数据/金融类合并资产负债表/'):
    for i in range(len(files)) : 
        if files[i][0:1] == '6' : 
            files[i] = files[i][0:7] + 'XSHG'
        else : 
            files[i] = files[i][0:7] + 'XSHE'   
for stock in files:
    q = query(finance.FINANCE_INCOME_STATEMENT).filter(
        finance.FINANCE_INCOME_STATEMENT.code == stock,
        finance.FINANCE_INCOME_STATEMENT.pub_date >= '2010-01-01',
        finance.FINANCE_INCOME_STATEMENT.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/金融类合并利润表/' + stock + '.csv', encoding='gbk')


# In[7]:


# 现金流量表
for stock in stock_list:
    q = query(finance.STK_CASHFLOW_STATEMENT).filter(
        finance.STK_CASHFLOW_STATEMENT.code == stock,
        finance.STK_CASHFLOW_STATEMENT.pub_date >= '2010-01-01',
        finance.STK_CASHFLOW_STATEMENT.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/合并现金流量表/' + stock + '.csv', encoding='gbk')


# In[14]:


# 金融类现金流量表
for stock in files:
    q = query(finance.FINANCE_CASHFLOW_STATEMENT).filter(
        finance.FINANCE_CASHFLOW_STATEMENT.code == stock,
        finance.FINANCE_CASHFLOW_STATEMENT.pub_date >= '2010-01-01',
        finance.FINANCE_CASHFLOW_STATEMENT.report_type == 0)
    df = finance.run_query(q)
    stock = stock[:6]
    df.to_csv('dataset/A股各公司财务数据/金融类合并现金流量表/' + stock + '.csv', encoding='gbk')


# In[ ]:


## 获取财务指标和估值数据


# In[5]:


# 生成季度list
date_season = []
for year in range(2010,2021):
    for season in [1,2,3,4] : 
        date_season.append(str(year) + 'q' + str(season))
date_season = date_season + ['2021q1']


# In[6]:


# 对每个季度，下载财务指标
for season in date_season : 
    df = get_fundamentals(query(indicator),statDate=season)

    # save
    df.sort_values(by='code')
    df.to_csv('dataset/A股各公司财务数据/财务指标数据/' + season + '.csv', encoding='gbk')


# In[4]:


# 对每个季度，下载估值数据
for season in date_season : 
    df = get_fundamentals(query(valuation),statDate=season)
    
    # save
    df.sort_values(by='code')
    df.to_csv('dataset/A股各公司财务数据/估值数据/' + season + '.csv', encoding='gbk')


# In[ ]:


## 获取交易日历


# In[3]:


# 获取交易日历
trading_date_arr = get_trade_days(start_date="2009-01-10",end_date="2021-07-10")
pd.DataFrame(trading_date_arr).to_csv('trading_date.csv')


# In[ ]:





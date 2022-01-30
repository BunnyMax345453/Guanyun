
# -*- coding: utf-8 -*-

# CrossSection
# 
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

class CrossSection():
    def __init__(self, base_data):
        '''
        func: 对输入的数据df以市值为权重进行WLS回归，输出因子收益和特殊收益
        input: base_data: DataFrame(columns: date, stocknames, capital, ret, industry_factors and style_factors)
        output: 因子收益factor_ret, 特殊收益specific_ret, R2
        '''
        self.data = base_data
        self.stocknames = list(base_data.stocknames)          #股票名
        self.date = list(base_data.date)[0]                   #日期
        self.capital = base_data.capital.values               #市值
        self.ret = base_data.ret.values                       #t+1期收益率
        self.W = np.sqrt(self.capital) / sum(np.sqrt(self.capital))   #加权最小二乘法的权重
        
        print('\rCross Section Regression, ' + 'Date: ' + self.date  + ', ', end = '')
        WLS = LinearRegression(fit_intercept=False)
        WLS.fit(self.data.iloc[:,4:], self.data.iloc[:,3], sample_weight = self.W)
        
        factor_ret = WLS.coef_                    #纯因子收益
        specific_ret = self.ret - WLS.predict(self.data.iloc[:,4:])             #个股特异性收益
        R2 = 1 - np.var(specific_ret) / np.var(self.ret)                            #R square
        
        return((factor_ret, specific_ret, R2))
    
    

class MFM():
    '''
    data: DataFrame
    column1: date
    colunm2: stocknames
    colunm3: capital
    column4: ret
    style_factors: DataFrame
    industry_factors: DataFrame
    '''
    
    def __init__(self, data, P, Q):
        '''
        func: 初始化
        input: data: DataFrame(columns: date, stocknames, capital, ret, industry_factors and style_factors)
            P: industry_factors 数量
            Q: style_factors 风格因子数量
        output: 无
        '''
        self.Q = Q                                                           #风格因子数
        self.P = P                                                           #行业因子数
        self.dates = pd.to_datetime(data.date.values)                        #日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   #排序后的日期
        self.T = len(self.sorted_dates)                                      #期数
        self.data = data                                                     #数据
        self.columns = ['country']                                           #因子名
        self.columns.extend((list(data.columns[4:])))
        
        self.last_capital = None                                             #最后一期的市值 
        self.factor_ret = None                                               #因子收益
        self.specific_ret = None                                             #特异性收益
        self.R2 = None                                                       #R2
        
        self.Newey_West_cov = None                        #逐时间点进行Newey West调整后的因子协方差矩阵
        self.eigen_risk_adj_cov = None                    #逐时间点进行Eigenfactor Risk调整后的因子协方差矩阵
        self.vol_regime_adj_cov = None                    #逐时间点进行Volatility Regime调整后的因子协方差矩阵
    
    
    def reg_by_time(self):
        '''
        func: 调用之前的CrossSection类，逐时间点进行横截面多因子回归
        output: 因子收益factor_ret, 特殊收益specific_ret, R2
        '''
        factor_ret = []
        R2 = []
        specific_ret = []
        
        print('===================================逐时间点进行横截面多因子回归===================================')       
        for t in range(self.T):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            cs = CrossSection(data_by_time)
            factor_ret_t, specific_ret_t , R2_t = cs.reg()
            
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            specific_ret.append(specific_ret_t)
            R2.append(R2_t)
            self.last_capital = cs.capital
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        specific_ret = pd.DataFrame(specific_ret, columns = cs.stocknames, index = self.sorted_dates)
        self.factor_ret = factor_ret                                               #因子收益
        self.specific_ret = specific_ret                                           #特异性收益
        self.R2 = R2                                                               #R2
        return((factor_ret, specific_ret, R2))


    def Newey_West_by_time(self, q = 2, tao = 252, content = 'factor_ret'):
        '''
        func: 逐时间点计算协方差并进行Newey West调整
        input: q: 假设因子收益为q阶MA过程
            tao: 算协方差时的半衰期
            content: 区别开因子收益和特殊收益
        output: Newey West调整后的df
        '''
        if content == 'factor_ret' : 
            if self.factor_ret is None:
                raise Exception('please run reg_by_time to get factor returns first')

            Newey_West_cov = []
            print('\n\n===================================逐时间点进行Newey West调整=================================')    
            for t in range(1,self.T+1):
                try:
                    Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao))
                except:
                    Newey_West_cov.append(pd.DataFrame())

                progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])

            self.Newey_West_cov = Newey_West_cov
            return(Newey_West_cov)
        if content == 'specific_ret' : 
            if self.specific_ret is None:
                raise Exception('please run reg_by_time to get specific returns first')

            Newey_West_cov = []
            print('\n\n===================================逐时间点进行Newey West调整=================================')    
            for t in range(1,self.T+1):
                try:
                    Newey_West_cov.append(Newey_West(self.specific_ret[:t], q=5, tao=90))
                except:
                    Newey_West_cov.append(pd.DataFrame())

                progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])

            self.Newey_West_cov = Newey_West_cov
            return(Newey_West_cov)
    
    
    def eigen_risk_adj_by_time(self, M = 100, scale_coef = 1.4):
        '''
        func: 逐时间点进行Eigenfactor Risk Adjustment
        input: M: 模拟次数
            scale_coef: scale coefficient for bias(研报建议为1.4)
        output: Eigenfactor Risk Adjustment后的df
        '''
        
        if self.Newey_West_cov is None:
            raise Exception('please run Newey_West_by_time to get factor return covariances after Newey West adjustment first')        
        
        eigen_risk_adj_cov = []
        print('\n\n===================================逐时间点进行Eigenfactor Risk调整=================================')    
        for t in range(self.T):
            try:
                eigen_risk_adj_cov.append(eigen_risk_adj(self.Newey_West_cov[t], self.T, M, scale_coef))
            except:
                eigen_risk_adj_cov.append(pd.DataFrame())
            
            progressbar(t+1, self.T, '   date: ' + str(self.sorted_dates[t])[:10])
        
        self.eigen_risk_adj_cov = eigen_risk_adj_cov
        return(eigen_risk_adj_cov)
        
        
    
    def vol_regime_adj_by_time(self, tao = 42, h = 252, content = 'factor_ret', data_cov = None):
        '''
        func: 对矩阵进行Volatility Regime Adjustment
        input: tao: Volatility Regime Adjustment的半衰期
            h: 回测天数
            content: 区别开因子收益和特殊收益
            data_cov: 在特殊因子调整时需要用到(content = 'specific_ret')，否则为None
        '''
        if content == 'factor_ret':        
            if self.eigen_risk_adj_cov is None:
                raise Exception('please run eigen_risk_adj_by_time to get factor return covariances after eigenfactor risk adjustment first')        


            K = len(self.eigen_risk_adj_cov[-1])
            res_var = list()
            for t in range(self.T):
                res_var_i = np.diag(self.eigen_risk_adj_cov[t])
                if len(res_var_i)==0:
                    res_var_i = np.array(K*[np.nan])
                res_var.append(res_var_i)

            res_var = np.array(res_var)

            B = np.sqrt(np.mean(self.factor_ret**2 / res_var, axis = 1))      #截面上的bias统计量
            weights = 0.5**(np.arange(h-1,-1,-1)/tao)                            #指数衰减权重 # new : self.T : h


            lamb = []
            vol_regime_adj_cov = []
            print('\n\n==================================逐时间点进行Volatility Regime调整================================') 
            for t in [self.T]:
                #取除无效的行
                if t <= 252 : 
                    okidx = pd.isna(res_var[:t]).sum(axis = 1) == 0 
                    okweights = weights[:t][okidx] / sum(weights[:t][okidx])
                    fvm = np.sqrt(sum(okweights * B.values[:t][okidx]**2))   #factor volatility multiplier
                else : 
                    okidx = pd.isna(res_var[t-252:t]).sum(axis = 1) == 0 
                    okweights = weights[:][okidx] / sum(weights[:][okidx])
                    fvm = np.sqrt(sum(okweights * B.values[t-252:t][okidx]**2))   #factor volatility multiplier

                lamb.append(fvm)  
                vol_regime_adj_cov.append(self.eigen_risk_adj_cov[t-1] * fvm**2)
                progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])

            self.vol_regime_adj_cov = vol_regime_adj_cov
            return((vol_regime_adj_cov, lamb))
        
        if content == 'specific_ret':
            res_var = list()
            weight_cap = list()
            for t in range(self.T):
                data_K = self.data[self.data['date'] == self.sorted_dates[t].strftime("%Y-%m-%d")]
                K = len(data_K)
                
                res_var_i = data_cov[t]
                weight_cap_i = data_K['capital']
                weight_cap_sum = data_K['capital'].sum()
                if len(res_var_i)==0:
                    res_var_i = np.array(K*[np.nan])
                res_var.append(res_var_i)
                weight_cap.append(weight_cap_i/weight_cap_sum)
            res_var = np.array(res_var)
            weight_cap = np.array(weight_cap)
            
            B = np.sqrt(np.sum(self.specific_ret**2 / res_var * weight_cap, axis = 1))     #截面上的bias统计量
            B[np.isnan(B)] = 0
            weights = 0.5**(np.arange(h-1,-1,-1)/tao)                            #指数衰减权重 # new : self.T : h

            lamb = []
            vol_regime_adj_cov = []
            print('\n\n==================================逐时间点进行Volatility Regime调整================================') 
            for t in [self.T]:
                #取除无效的行
                if t < 253 : 
                    lamb.append([])  
                    vol_regime_adj_cov.append(pd.Series())
                else : 
                    okidx = pd.isna(res_var[t-252:t]).sum(axis = 1) == 0 
                    okweights = weights[:][okidx] / sum(weights[:][okidx])
                    fvm = np.sqrt(np.sum(okweights * B.values[t-252:t][okidx]**2))   #factor volatility multiplier
                    lamb.append(fvm)  
                    vol_regime_adj_cov.append(data_cov[t-1] * fvm)
                progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])

            self.vol_regime_adj_cov = vol_regime_adj_cov
            return((vol_regime_adj_cov, lamb))

        

import matplotlib.pyplot as plt
import math



def Newey_West(ret, q = 2, tao = 90, h = 252):
    '''
    func: Newey_West方差调整
        时序上存在相关性时，使用Newey_West调整协方差估计
    factor_ret: DataFrame, 行为时间，列为因子收益
    q: 假设因子收益为q阶MA过程
    tao: 算协方差时的半衰期
    '''
    from functools import reduce
    from statsmodels.stats.weightstats import DescrStatsW 
    
    T = ret.shape[0]           #时序长度
    K = ret.shape[1]           #因子数/股票数
    if T <= q or T < h:
        raise Exception("T <= q or T < h") # new: add h
    ret = ret.iloc[-h:,:] # new: add h
    names = ret.columns    
    weights = 0.5**(np.arange(h-1,-1,-1)/tao)   #指数衰减权重 # new: T to h
    weights = weights / sum(weights)
    
    w_stats = DescrStatsW(ret, weights)
    ret = ret - w_stats.mean
    
    ret = np.matrix(ret.values)
    Gamma0 = [weights[t] * ret[t].T  @ ret[t] for t in range(h)] # new: T to h
    Gamma0 = reduce(np.add, Gamma0)
    
    
    V = Gamma0             #调整后的协方差矩阵
    for i in range(1,q+1):
        Gammai = [weights[i+t] * ret[t].T  @ ret[i+t] for t in range(h-i)]  # new: T to h
        Gammai = reduce(np.add, Gammai)
        V = V + (1 - i/(1+q)) * (Gammai + Gammai.T)
    
    return(pd.DataFrame(V, columns = names, index = names))
    
    
    

def eigen_risk_adj(covmat, T = 100, M = 100, scale_coef = 1.2):
    '''
    func: Eigenfactor Risk Adjustment
    input: T: 序列长度
        M: 模拟次数
        scale_coef: scale coefficient for bias
    '''
    F0 = covmat
    K = covmat.shape[0]
    D0,U0 = np.linalg.eig(F0)      #特征值分解; D0是特征因子组合的方差; U0是特征因子组合中各因子权重; F0是因子协方差方差
    #F0 = U0 @ D0 @ U0.T    D0 = U0.T @ F0 @ U0  
    
#     if not all(D0>=0):         #检验正定性
#         raise('covariance is not symmetric positive-semidefinite')
   
    v = []  #bias
    for m in range(M):
        ## 模拟因子协方差矩阵
        np.random.seed(m+1)
        bm = np.random.multivariate_normal(mean = K*[0], cov = np.diag(D0), size = T).T  #特征因子组合的收益
        fm = U0 @ bm       #反变换得到各个因子的收益
        Fm = np.cov(fm)    #模拟得到的因子协方差矩阵

        ##对模拟的因子协方差矩阵进行特征分解
        Dm,Um = np.linalg.eig(Fm)   # Um.T @ Fm @ Um 
    
        ##替换Fm为F0
        Dm_hat = Um.T @ F0 @ Um 

        v.append(np.diagonal(Dm_hat) / Dm)

    v = np.sqrt(np.mean(np.array(v), axis = 0))
    v = scale_coef * (v-1) + 1
    
    
    D0_hat = np.diag(v**2) * np.diag(D0)  #调整对角线
    F0_hat = U0 @ D0_hat @ U0.T           #调整后的因子协方差矩阵
    return(pd.DataFrame(F0_hat, columns = covmat.columns, index = covmat.columns))

    


def eigenfactor_bias_stat(cov, ret, predlen = 1):
    '''
    func: 计算特征因子组合的bias统计量
    '''
    #bias stat
    b = []
    for i in range(len(cov)-predlen):
        try:
            D, U = np.linalg.eig(cov[i])                              #特征分解, U的每一列就是特征因子组合的权重
            U = U / np.sum(U, axis = 0)                               #将权重标准化到1
            sigma = np.sqrt(predlen * np.diag(U.T @ cov[i] @ U))      #特征因子组合的波动率
            retlen = (ret.values[(i+1):(i+predlen+1)] + 1).prod(axis=0) - 1
            r = U.T @ retlen                                          #特征因子组合的收益率
            b.append(r / sigma)
        except:
            pass
    
    b = np.array(b)
    bias_stat = np.std(b, axis = 0)
    plt.plot(bias_stat)
    return(bias_stat)
    


    
def progressbar(cur, total, txt):
    '''
    func: 显示进度条
    '''
    percent = '{:.2%}'.format(cur / total)
    print("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent) + txt, end = '')
    



def group_mean_std(x):
    '''
    func: 计算一组的加权平均和波动率
    '''
    m =sum(x.volatility*x.capital) / sum(x.capital)
    s = np.sqrt(np.mean((x.volatility - m)**2))
    return([m, s])


def shrink(x, group_weight_mean, q):
    '''
    func: 计算shrink估计量
    '''
    a = q * np.abs(x['volatility'] - group_weight_mean[x['group']][0])
    b =  group_weight_mean[x['group']][1]
    v = a / (a + b)     #收缩强度
    SH_est = v * group_weight_mean[x['group']][0] + (1-v) * np.abs(x['volatility'])    #贝叶斯收缩估计量
    return(SH_est)
    

def bayes_shrink(volatility, capital, ngroup = 10, q = 1):
    '''
    func: 使用市值对特异性收益率波动率进行贝叶斯收缩，以保证波动率估计在样本外的持续性
    input: volatility: 波动率
        capital: 市值
        ngroup: 划分的组数
        q: shrinkage parameter
    '''
    group = pd.qcut(capital,  10, [1,2,3,4,5,6,7,8,9,10]).values   #按照市值分为10组
    data = pd.DataFrame(np.array([volatility, capital, group]).T, columns = ['volatility', 'capital', 'group'])
    #分组计算加权平均
    grouped = data.groupby('group')
    group_weight_mean = grouped.apply(group_mean_std)
    
    SH_est = data.apply(shrink, axis = 1, args = (group_weight_mean, q))   #贝叶斯收缩估计量 
    SH_est.index = capital.index
    return(SH_est)


def structural_model(i, data_t, sigma_NW, h = 252):
    """
    func: 对data进行结构化调整
    ret = specific_ret
    """
    # （1）首先计算每只股票特异收益的稳健标准差𝜎̃u：
    sigma_u = (1/1.35) * ( ret.iloc[-h:,:].quantile(3/4) - ret.iloc[-h:,:].quantile(1/4) )
    # 计算𝜎𝑢,𝑒𝑞，特异收益的样本标准差
    sigma_ueq = ret.std(axis=0,ddof=1)
    # 计算特异收益的肥尾程度指标𝑍u。若𝑍𝑢值过大，则说明该特异收益序列中存在异常值。
    z_u = abs( (sigma_ueq - sigma_u) / sigma_u )

    t = np.exp(1 - z_u)
    t[t<=0] = 0  # max(0, np.exp(1 - z_u))
    t[t>=1] = 1  # min(1,x)
    # 引入协调参数γ
    gamma = (min(1,max(0, (h-60)/120))) * t
    list_i = list(set(gamma[gamma==1].index.to_list()).intersection(set(data_t['stocknames'].to_list())).intersection(set(sigma_NW.index.to_list()))) # 取交集，后续删除

    sigma_NW = sigma_NW
    sigma_TR = np.diag(sigma_NW)
    sigma_TR = pd.Series(sigma_TR,index = sigma_NW.index)
    sigma_TR = sigma_TR[list_i]

    reg_t = data_t.set_index('stocknames')
    reg_t = reg_t.loc[list_i,:].iloc[:,3:]
    reg_t['sigma_TR'] = np.log(sigma_TR)  # np.ln(sigma_TR)

    # 改为WLS回归
    W_df = data[data['date'] == data.iloc[-1]['date']].set_index('stocknames')
    W_df = W_df.loc[list_i]
    capital = W_df.capital.values             #市值
    W = np.sqrt(capital) / sum(np.sqrt(capital))   #加权最小二乘法的权重
    WLS = LinearRegression()
    WLS.fit((reg_t.iloc[:,:-1]), (reg_t.iloc[:,-1]), sample_weight = W)
#     smf_X = np.matrix(reg_t.iloc[:,:-1])
#     smf_y = np.matrix(reg_t.iloc[:,-1]).T
#     b_k = (smf_X.T @ smf_X).I @ smf_X.T @ smf_y
    b_k = WLS.coef_
    b_k = np.array(b_k.reshape(1,37))[0]
    data_t = data_t.iloc[:,4:]
    sigma_STR = (data_t.apply(lambda x : np.exp((x * b_k).sum()),axis = 1)) * 1.05
    res = np.array(gamma) * np.diag(sigma_NW) + np.array(1 - gamma) * np.array(sigma_STR)
    res = pd.Series(res, index = gamma.index)
    return res


def min_vol_fac_port(V, X, factor_k):
    """
    func: 输出最小波动组合
    input: V: 股票收益的协方差矩阵（N×N 维） dataframe
        X: 股票因子暴露矩阵（N×K维） dataframe 
        X_k: 表示股票在因子k上的暴露度（N×1维） dataframe 
        factor_k: 因子k list
    """
    
    stks_i = V.index
    X = X.loc[stks_i,:]
    import numpy.linalg as lg
    X_k = X[factor_k]
    V = np.matrix(V.values)
    X_k = np.matrix(X_k.values)

    V[np.isnan(V)] = 0

    res = (V.I @ X_k) / (X_k.T @ V.I @ X_k).diagonal()
    return(pd.DataFrame(res, columns = factor_k, index = stks_i))

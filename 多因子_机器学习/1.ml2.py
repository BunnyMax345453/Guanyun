import numpy as np
import pandas as pd
from datetime import timedelta
import scipy
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

from sklearn import linear_model, svm, metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from keras.utils import multi_gpu_model  
from tensorflow.keras import regularizers 
from tensorflow.keras.preprocessing import sequence  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Flatten, Reshape , Dropout , Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

def lgb_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters):
    """
    function: LGB决策树分类
    input: 训练集，验证集，检测集
    output: 检测集预测结果(array)
    """
    event_rate=y_train.mean()
    event_cut_off=1-event_rate 

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_valid, X_test = sc.transform(X_valid), sc.transform(X_test)
    d_train = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    test_data = lgb.Dataset(X_test, label=y_test)
    # 给定参数搜索范围：list or distribution
    param_dist = {"learning_rate": scipy.stats.expon(scale=0.1),                     #给定list
              'max_depth': np.random.randint(3,9,10),          #给定distribution
              "num_leaves": 2**np.random.randint(4,7,10),     #给定distribution
#               'min_data_in_leaf': np.random.randint(8,16),
#               'feature_fraction': np.random.uniform(0.5,0.9),
#               'bagging_fraction': np.random.uniform(0.7,0.9)
                 }
    # 用RandomSearch+CV选取超参数
    clf = lgb.LGBMClassifier()
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                           n_iter=n_iter_search, cv=5, iid=False)
    logreg=random_search.fit(X_train, y_train)
    prepro = random_search.predict_proba(X_test)
    return prepro[:,0]

def network(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters):
    """
    function: 单隐藏层神经网络预测
    input: 训练集，验证集，检测集
    output: 检测集预测结果(array)
    """
    # 训练集归一化  
    min_max_scaler = MinMaxScaler()  
    min_max_scaler.fit(X_train)  
    X_train = min_max_scaler.transform(X_train)  
    min_max_scaler.fit(pd.DataFrame(y_train))  
    y_train = min_max_scaler.transform(pd.DataFrame(y_train)).reshape(-1)
    # 验证集归一化  
    min_max_scaler.fit(X_valid)  
    X_valid = min_max_scaler.transform(X_valid)  
    min_max_scaler.fit(pd.DataFrame(y_valid))  
    y_valid = min_max_scaler.transform(pd.DataFrame(y_valid)).reshape(-1)
    # 验证集归一化  
    min_max_scaler.fit(X_test)  
    X_test = min_max_scaler.transform(X_test)  
    ### 搭建网络
    # 单CPU or GPU版本，若有GPU则自动切换  
    model = Sequential()  # 初始化，很重要！

    model.add(Dense(units = parameters['units'],   # 输出大小  
                    activation='relu',  # 激励函数  
                    input_shape=(parameters['input'],)  # 输入大小, 也就是列的大小  
                   )  
             )  
    model.add(Dense(units = 1,     
                    activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数    
                   )  
             )  
    ### 训练
    model.compile(loss=parameters['loss'],  # 损失均方误差  
              optimizer=parameters['optimizer'],  # 优化器  
#               metrics=['r2_score']
                 )  
    history = model.fit(X_train, y_train,  
              epochs=parameters['epochs'],  # 迭代次数  
              batch_size=parameters['batch_size'],  # 每次用来梯度下降的批处理数据大小  
              verbose=parameters['verbose'],  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch  
              validation_data = (X_valid, y_valid)  # 验证集  
            )
    return model.predict(X_test)


def lr_func(x_train, x_valid, x_test, y_train, y_valid, y_test, parameters):
    """
    function: 逻辑回归
    input: 训练集，验证集，检测集
    output: 检测集预测结果(array)
    """
    # 标准化特征值
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    # 给定参数搜索范围：list or distribution
    param_dist = {"C": scipy.stats.expon(scale=0.1),                     #给定list
              'fit_intercept': [True,False],          #给定distribution
              "tol": np.random.rand(20)*0.001,     #给定distribution
             }
    # 用RandomSearch+CV选取超参数
    n_iter_search = 20
    random_search = RandomizedSearchCV(linear_model.LogisticRegression(class_weight= 'balanced',solver= 'liblinear'), 
                           n_jobs = 5,
                           param_distributions=param_dist,
                           n_iter=n_iter_search, cv=5, iid=False)
    logreg=random_search.fit(x_train_std, y_train)
    # 预测
    prepro = logreg.predict_proba(x_test_std)
    return prepro[:,1]


def cnn_func(x_train, x_valid, x_test, y_train, y_valid, y_test, parameters):
    """
    function: 卷积神经网络预测模型
    input: 训练集，验证集，检测集
    output: 检测集预测结果(array)
    """
    ### 搭建网络
    model = Sequential()
    model.add(Conv1D(20, 4, padding='same',activation='relu',
                    input_shape=(parameters['input'],1))) # 1D cnn/ padding不改变维度
    model.add(MaxPooling1D(2))  # 池化层
    model.add(Flatten())  # 摊平为1D
    model.add(Dense(20))  # 20个神经元的全连接层
    model.add(Dropout(0.2))  #  防止过拟合，随即冻结20%
    model.add(Activation('relu'))
    model.add(Dense(1)) # 输出层，只有1D
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer=SGD(lr=0.2), metrics=['accuracy'])
    # 训练集归一化  
    min_max_scaler = MinMaxScaler()  
    min_max_scaler.fit(x_train)  
    x_train = min_max_scaler.transform(x_train)  
    min_max_scaler.fit(pd.DataFrame(y_train))  
    y_train = min_max_scaler.transform(pd.DataFrame(y_train)).reshape(-1)
    # 验证集归一化  
    min_max_scaler.fit(x_valid)  
    x_valid = min_max_scaler.transform(x_valid)  
    min_max_scaler.fit(pd.DataFrame(y_valid))  
    y_valid = min_max_scaler.transform(pd.DataFrame(y_valid)).reshape(-1)
    # 验证集归一化  
    min_max_scaler.fit(x_test)  
    x_test = min_max_scaler.transform(x_test)  
    
    tf.config.run_functions_eagerly(True)
    model.fit(x_train[:,:,np.newaxis], y_train, 
              epochs=50,   # 迭代次数  
              batch_size=1024,   # 每次用来梯度下降的批处理数据大小  
              verbose=0,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch  
              validation_data = (x_valid[:,:,np.newaxis], y_valid))   # 验证集  

    return model.predict(x_test[:,:,np.newaxis])


def linear_reg_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters):
    """
    function: 线性回归
    input: 训练集，验证集，检测集
    output: 检测集预测结果(array)
    """
    # 标准化特征值
    sc = StandardScaler()
    sc.fit(X_train)
    x_train_std = sc.transform(X_train)
    x_test_std = sc.transform(X_test)
    # 训练逻辑回归模型
    logreg = linear_model.LinearRegression(fit_intercept = False,n_jobs=5)
    logreg.fit(x_train_std, y_train)
    # 预测
    prepro = logreg.predict_proba(x_test_std)
    return prepro[:,1]


def ml(model, X, y, parameters, rolling=5, delay=1):
    """
    function: 为不同模型的选择分配function
    input: 模型名称，自变量，因变量，参数，需要rolling的天数，需要滞后的天数
    output: 检测集预测结果(array)
    """
    X_train = np.array(X[0])
    X_valid = np.array(X[1])
    X_test = np.array(X[2])
    y_train = y[0]
    y_valid = y[1]
    y_test = y[2]
    postfix = '_rolling'+str(rolling)+'_delay'+str(delay)
    if model in ['lgb','lr',]:
        y_name = 'relative_target' + postfix
        y_train = np.array(y_train[y_name])
        y_valid = np.array(y_valid[y_name])
        y_test = np.array(y_test[y_name])
        if model == 'lgb':
            return lgb_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters)
        elif model == 'lr':
            return lr_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters)
    elif model in ['nn','linear_reg','cnn',]:
        y_name = 'return' + postfix
        y_train = np.array(y_train[y_name])
        y_valid = np.array(y_valid[y_name])
        y_test = np.array(y_test[y_name])
        if model == 'nn':
            return network(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters)
        elif model == 'linear_reg' : 
            return linear_reg_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters)
        elif model == 'cnn':
            return cnn_func(X_train, X_valid, X_test, y_train, y_valid, y_test, parameters)
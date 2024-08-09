# -*- coding: utf-8 -*-
# https://daniel820710.medium.com/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ----- functions -----
def ReadData(): # read data
    data = pd.read_csv("output.csv")
    return data

def DataClean(data):
    # 删除列名中的所有空格
    data.columns = [col.replace(' ', '') for col in data.columns]

    # one hot encode
    one_hot_columns = []
    """
    onehot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoding = onehot_encoder.fit_transform(data[one_hot_columns])
    one_hot_encoding = pd.DataFrame(one_hot_encoding, columns=onehot_encoder.get_feature_names_out(one_hot_columns))
    data, one_hot_encoding = data.reset_index(drop=True), one_hot_encoding.reset_index(drop=True)
    data = pd.concat([data, one_hot_encoding], axis=1)
    """

    # factor
    factor_columns = ['方向', '車輛1', '車輛2', '車輛3', '車輛4', '車輛5', '車輛6', '車輛7', '車輛8', '車輛9', '車輛10', '車輛11', '車輛12', '行政區域', '國道名稱']
    data.factor_names = []
    for factor in factor_columns:
        factor_codes, factor_names = pd.factorize(data[factor])
        data[factor] = factor_codes
        factor_names = [factor_names]
        data.factor_names.append(factor_names)
    data.factor_columns = factor_columns

    # drop
    drop_columns = ['事件排除_時', '事件排除_分', '經度', '緯度', '主線中斷註記', '肇事車輛', '死亡', '受傷', '翻覆事故註記', '施工事故註記', '危險物品車輛註記', '車輛起火註記', '冒煙車事故註記']
    data.drop(columns=one_hot_columns + drop_columns, inplace=True) # inplace >> 直接修改data 
    data = data.fillna(0)
    return data

def buildTrain(train, y, pastDay = 30, futureDay = 5): # lag, 使用過去30天訓練預測未來5天
    X_train, Y_train = [], []
    for i in range(train.shape[0] - futureDay - pastDay):
        X_train.append(np.array(train.iloc[i:i + pastDay]))
        Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay][y]))
    return np.array(X_train), np.array(Y_train)

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0]) # 依X列數量生成數列(編號)
    np.random.shuffle(randomList) # 打亂數列
    return X[randomList], Y[randomList]

def splitData(X, Y, rate): # 拆分訓練集
    X_train = X[int(X.shape[0] * rate):]
    Y_train = Y[int(Y.shape[0] * rate):]
    X_val = X[:int(X.shape[0] * rate)]
    Y_val = Y[:int(Y.shape[0] * rate)]
    return X_train, Y_train, X_val, Y_val

def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(units = 10, input_shape=(shape[1], shape[2]), return_sequences = True))
    # output shape: (1, 1)
    model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
    model.compile(loss = "mse", optimizer = "adam")
    model.summary()
    return model

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(units = 10, input_shape=(shape[1], shape[2])))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss = "mse", optimizer = "adam")
    model.summary()
    return model

def buildOneToManyModel(shape):
    model = Sequential()
    model.add(LSTM(units = 10, input_shape=(shape[1], shape[2])))
    # output shape: (5, 1)
    model.add(Dense(1))
    model.add(RepeatVector(5)) # 5 : 5天(次)
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(units = 10, input_shape=(shape[1], shape[2]), return_sequences = True))
    # output shape: (5, 1)
    model.add((Dense(1)))
    model.compile(loss = "mse", optimizer = "adam")
    model.summary()
    return model

# ----- data import -----
# read csv data
data = ReadData()

# data clean
data = DataClean(data)
data.to_csv('output1.csv')

X_train, Y_train = buildTrain(data, '處理分鐘', 1 , 1)
#X_train, Y_train = data.drop(columns=['處理分鐘']), data['處理分鐘']
X_train, Y_train = shuffle(X_train, Y_train)

X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

model = buildManyToOneModel(X_train.shape)

callback = EarlyStopping(monitor = "loss", 
                         patience = 10, 
                         verbose = 1, 
                         mode = "auto"
                         )

model.fit(X_train, 
          Y_train, 
          epochs=1000, 
          batch_size=128, 
          validation_data=(X_val, Y_val), 
          callbacks=[callback]
          )

# 預測
predicted = model.predict(X_val)

# 繪製結果
plt.plot(Y_val, label='True Value')
plt.plot(predicted, label='Predicted Value')
plt.title('Prediction vs True Value')
plt.xlabel('Sample Index')
plt.ylabel('處理分鐘')
plt.legend()
plt.show()









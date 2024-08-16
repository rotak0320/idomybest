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
    data_A = pd.read_excel("112年1-10月A1事故資料(113.01.12更新).xlsx")
    data_B = pd.read_excel("112年1-10月A2事故資料(113.02.06更新).xlsx")
    data_C = pd.read_excel("112年1-10月A3事故資料(113.02.06更新).xlsx")
    data = pd.concat([data_A, data_B, data_C], axis=0)
    return data

def DataClean(data):
    # 删除列名中的所有空格
    data.columns = [col.replace(' ', '') for col in data.columns]

    # one hot encode
    one_hot_columns = ['事故類別', '路線', '分局', '當事者屬(性)別']
    onehot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoding = onehot_encoder.fit_transform(data[one_hot_columns])
    one_hot_encoding = pd.DataFrame(one_hot_encoding, columns=onehot_encoder.get_feature_names_out(one_hot_columns))
    data, one_hot_encoding = data.reset_index(drop=True), one_hot_encoding.reset_index(drop=True)
    data = pd.concat([data, one_hot_encoding], axis=1)

    # factor
    factor_columns = ['車道線(側)名稱', '天候', '道路型態', '車道劃分設施-分向設施', '車道劃分設施-分向設施', '事故類型及型態2', '受傷程度', '保護裝備', '行動電話、電腦或其他相類功能裝置名稱', '當事者區分(大類別)', '當事者區分(類別)', '車輛用途', '飲酒情形名稱', '初步分析研判子類別-主要']
    data.factor_names = []
    for factor in factor_columns:
        factor_codes, factor_names = pd.factorize(data[factor])
        data[factor] = factor_codes
        factor_names = [factor_names]
        data.factor_names.append(factor_names)
    data.factor_columns = factor_columns

    # other
    data[['24小時內死亡人數', '2-30日內死亡人數']] = data[['24小時內死亡人數', '2-30日內死亡人數']].fillna(0)
    data['死亡'] = data['24小時內死亡人數'] + data['2-30日內死亡人數']

    # drop
    drop_columns = ['工務段', '向', '24小時內死亡人數', '2-30日內死亡人數', '道路類別', '道路照明設備(11207新增)', '事故位置', '路面狀況-路面鋪裝', '路面狀況-路面狀態', '路面狀況-路面缺陷', '道路障礙-障礙物', '道路障礙-視距', '號誌-號誌種類', '號誌-號誌動作', '車道劃分設施-分道設施-路面邊線', '事故類型及型態代碼', '肇事逃逸(是否肇逃)', '車道劃分設施-分道設施-快慢車道間', '車道劃分設施-分道設施-快車道或一般車道間', '縣市', '市區鄉鎮']
    data.drop(columns=one_hot_columns + drop_columns, inplace=True) # inplace >> 直接修改data 
    
    return data

# ----- data import -----
# read csv data
data = ReadData()

# data clean
data = DataClean(data)
# data.to_csv('output.csv')








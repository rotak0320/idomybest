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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ----- functions -----
def ReadData(): # read data
    data = pd.read_csv("output.csv")
    rain = pd.read_csv("雨量資料_2023.csv")
    return data, rain

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
    factor_columns = ['方向', '車輛1', '車輛2', '車輛3', '車輛4', '車輛5', '車輛6', '車輛7', '車輛8', '車輛9', '車輛10', '車輛11', '車輛12', '行政區域', '國道名稱', 'StationName', 'ID', 'Attribute', 'Location']
    data.factor_names = []
    for factor in factor_columns:
        factor_codes, factor_names = pd.factorize(data[factor])
        data[factor] = factor_codes
        factor_names = [factor_names]
        data.factor_names.append(factor_names)
    data.factor_columns = factor_columns

    data['死傷'] = data['死亡'] + data['受傷']

    # drop
    drop_columns = ['事件排除_時', '事件排除_分', '經度', '緯度', '主線中斷註記', '肇事車輛', '死亡', '受傷', '翻覆事故註記', '施工事故註記', '危險物品車輛註記', '車輛起火註記', '冒煙車事故註記', 'Unnamed:0', 'Date', 'year', 'month', 'day']
    data.drop(columns=one_hot_columns + drop_columns, inplace=True) # inplace >> 直接修改data 
    
    data['Precipitation'] = [0.0 if T == "T" else T for T in data['Precipitation']] # 雨量為T表示雨量不足0.5mm
    data.astype(float)
    data = data.fillna(0)
    return data

def ClassifyMinute(minute):
    if minute < 10:
        return 0
    elif minute < 17:
        return 1
    elif minute < 30:
        return 2
    elif minute < 60:
        return 3
    else:
        return 4

def normalize(train): # 歸一化（又稱正規化 Normalization）
  train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return train_norm

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
data, rain = ReadData()

# 鄉鎮市區資料合併
rain['year'], rain['month'], rain['day'] = zip(*[(date[:4], date[5:7], date[8:10]) for date in rain['Date']])
data['行政區域'] = [admin.split(',')[0] if isinstance(admin, str) else "NA" for admin in data['行政區域']]
data['月'] = data['月'].astype(int)
data['日'] = data['日'].astype(int)
data['行政區域'] = data['行政區域'].astype(str)
rain['month'] = rain['month'].astype(int)
rain['day'] = rain['day'].astype(int)
rain['Location'] = rain['Location'].astype(str)

"""
def extract_district(address):
    # 找到 "市" 或 "縣" 和 "區" 的位置
    city_index = address.find("市") # 直轄市、省轄市
    county_index = address.find("縣")
    country_index = address.find("鄉")
    town_index = address.find("鎮")
    county_city_index = address.find("市") # 縣轄市
    district_index = address.find("區")
    
    # 確保 "市" 或 "縣" 和 "鄉"、"鎮"、"市"、"區" 都存在，并且 "市" 或 "縣" 在 "鄉"、"鎮"、"市"、"區" 之前
    if county_index != -1:
        if country_index != -1 and county_index < country_index:
            name = address[county_index + 1:country_index + 1]
            return name
        elif town_index != -1 and county_index < town_index:
            name = address[county_index + 1:town_index + 1]
            return name
        elif district_index != -1 and county_index < district_index:
            name = address[county_index + 1:district_index + 1]
            return name
        elif county_city_index != -1 and county_index < county_city_index:
            name = address[county_index + 1:county_city_index + 1]
            return name
    elif city_index != -1:
        if district_index != -1 and city_index < district_index:
            name = address[city_index + 1:district_index + 1]
            return name
    return address  # 如果无法提取区名，则返回原字符串

rain['Location'] = [extract_district(addr) for addr in rain['Location']]

# 合併 DataFrame，依據 '行政區域'、'月' 和 '日' 匹配
merged_data = pd.merge(
    left=data,
    right=rain,
    left_on=['行政區域', '月', '日'],
    right_on=['Location', 'month', 'day'],
    how='left'
)
merged_data.to_csv('output2.csv')
"""

combine_row = []
# 座標相互計算
from scipy.spatial.distance import cdist
# 抓取相同日期
for row in range(len(data)):
    rain_date = rain[(rain['month'] == data.iloc[row]['月']) & (rain['day'] == data.iloc[row]['日'])]
    # 使用 cdist 計算 兩資料框 之間的歐式距離
    # 提取 '經度' 和 '緯度' 作為 numpy 陣列，並確保它是二維的
    data_coords = data.iloc[row][['經度', '緯度']].astype(float).values.reshape(1, -1)  # 轉換為 (1, 2) 形狀
    rain_coords = rain_date[['Longitude', 'Latitude']].astype(float).values  # 轉換為 (n, 2) 形狀
    # 計算歐式距離矩陣
    distance_matrix = cdist(data_coords, rain_coords, metric='euclidean')
    # 將距離矩陣轉換為 DataFrame 以便查看
    distance_df = pd.DataFrame(distance_matrix, columns=[f'rain_row_{j}' for j in range(len(rain_date[['Longitude', 'Latitude']]))])
    # 找到每列最小的索引位置
    min_index = np.argmin(distance_df, axis=1)
    # 資料依距離合併
    data_row_df = pd.DataFrame([data.iloc[row].values], columns=data.iloc[row].index)
    combine = pd.concat([data_row_df.reset_index(drop=True), rain.iloc[min_index].reset_index(drop=True)], axis=1)
    combine['distance'] = distance_df.iloc[0, min_index].values # 座標距離
    combine_row.append(combine)

# 將結果轉換為新的 DataFrame
name = data.columns.append(rain.columns) # colnames
name = name.append(pd.Index(['distence']))
shape = np.array(combine_row).shape # 形狀修改
data = pd.DataFrame(np.reshape(combine_row, (shape[0], shape[2]))).reset_index(drop=True)
data.columns = name
data.to_csv('output3.csv')

# data clean
data = DataClean(data)
data['處理分鐘'] = data['處理分鐘'].apply(ClassifyMinute)
#data = normalize(data)
data.to_csv('output1.csv')

X_train, Y_train = buildTrain(data, '處理分鐘', 100 , 1) # 過去100筆預測未來1筆
X_train, Y_train = shuffle(X_train, Y_train)

X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

model = buildManyToOneModel(X_train.shape)

callback = EarlyStopping(monitor = "loss", 
                         patience = 100, 
                         verbose = 1, 
                         mode = "auto"
                         )

X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
Y_val = Y_val.astype(np.float32)

model.fit(X_train, 
          Y_train, 
          epochs=10000, 
          batch_size=128, 
          validation_data=(X_val, Y_val), 
          callbacks=[callback]
          )

# 預測
predicted = model.predict(X_val)
def SortResult(data):
    weight = (max(predicted) - min(predicted)) / 5
    min1, max1 = min(predicted), max(predicted)
    if data < min1 + weight:
        return 0
    elif data < min1 + 2 * weight:
        return 1
    elif data < min1 + 3 * weight:
        return 2
    elif data < max1:
        return 3
    else:
        return 4
predicted = np.array([SortResult(value) for value in predicted])

# 繪製結果
plt.plot(Y_val, label='True Value')
plt.plot(predicted, label='Predicted Value')
plt.title('Prediction vs True Value')
plt.xlabel('Sample Index')
plt.ylabel('處理分鐘')
plt.legend()
plt.show()

# 生成混淆矩陣
cm = confusion_matrix(Y_val, predicted)

# 繪製混淆矩陣
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()








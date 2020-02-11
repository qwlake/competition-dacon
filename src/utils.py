import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

# import logging
# logger = logging.getLogger("my")
# logger.setLevel(logging.INFO)

def data_loader_v2(file_name, folder='', train_label=None, event_time=10, nrows=60):
    file_id = int(file_name.split('.')[0]) # file id만 불러오기
    df = pd.read_csv(folder+file_name, index_col=0, nrows=nrows) # 파일 읽어오기
    df = df.replace('.*', 0, regex=True).fillna(0) # 모든 문자열과 NA값을 0으로 대체
    df = df.loc[event_time:] # event_time 이후의 row들만 가지고 오기
    df.index = np.repeat(file_id, len(df)) # row 인덱스를 file id로 덮어 씌우기 
    if type(train_label) != type(None):
        label = train_label.loc[file_id]['label'] 
        df['label'] = np.repeat(label, len(df)) #train set일 경우 라벨 추가하기 
#     print(df.shape)
    return df

def data_loader_v3(file_name, folder='', train_label=None, event_time=10, nrows=60):
    file_id = int(file_name.split('.')[0]) # file id만 불러오기
    data = pd.read_csv(folder+file_name, index_col=0, nrows=nrows) # 파일 읽어오기
    data = data.replace('.*', 0, regex=True).fillna(0) # 모든 문자열과 NA값을 0으로 대체
    data = data.loc[event_time:] # event_time 이후의 row들만 가지고 오기
    data.index = np.repeat(file_id, len(data)) # row 인덱스를 file id로 덮어 씌우기 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    data = torch.as_tensor(data, dtype=torch.float32)
    if type(train_label) != type(None):
        label = train_label.loc[file_id]['label']
        train_label = np.array([label]*len(data))
        train_label = torch.as_tensor(train_label, dtype=torch.float32)
        train_label = torch.unsqueeze(train_label, 1)
    return data, train_label

def data_loader_v4(file_name, folder='', train_label=None, event_time=10, nrows=60, batch=1):    
    file_id = int(file_name.split('.')[0]) # file id만 불러오기
    data = pd.read_csv(folder+file_name, index_col=0, nrows=nrows) # 파일 읽어오기
    data = data.replace('.*', 0, regex=True).fillna(0) # 모든 문자열과 NA값을 0으로 대체
    data = data.loc[event_time:] # event_time 이후의 row들만 가지고 오기
    data = data.loc[np.arange(len(data))%batch==0]
    data.index = np.repeat(file_id, len(data)) # row 인덱스를 file id로 덮어 씌우기 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    data = torch.as_tensor(data, dtype=torch.float32)
    if type(train_label) != type(None):
        label = train_label.loc[file_id]['label']
        train_label = np.array([label]*len(data))
        train_label = torch.as_tensor(train_label, dtype=torch.float32)
        train_label = torch.unsqueeze(train_label, 1)
    return data, train_label

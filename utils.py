import pandas as pd
import numpy as np
import requests

def add_variables(data):
    # 지난 시가 대비 시가 비율
    data['open_lastopen_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastopen_ratio'] = (data['open'][1:].values - data['open'][:-1].values) / data['open'][
                                                                                                 :-1].values
    # 지난 고가 대비 고가 비율
    data['high_lasthigh_ratio'] = np.zeros(len(data))
    data.loc[1:, 'high_lasthigh_ratio'] = (data['high'][1:].values - data['high'][:-1].values) / data['high'][
                                                                                                 :-1].values
    # 지난 저가 대비 고가 비율
    data['low_lastlow_ratio'] = np.zeros(len(data))
    data.loc[1:, 'low_lastlow_ratio'] = (data['low'][1:].values - data['low'][:-1].values) / data['low'][:-1].values
    # 지난 종가 대비 종가비율
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = (data['close'][1:].values - data['close'][:-1].values) / data['close'][
                                                                                                     :-1].values
    # 지난 거래량 대비 거래량 비율
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (data['volume'][1:].values - data['volume'][:-1].values) / data['volume'][
                                                                                                         :-1].values

    # 고가 대비 종가 비율
    data['high_colse_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    # 저가 대비 종가 비율
    data['low_colse_ratio'] = (data['low'].values - data['close'].values) / data['close'].values

    windows = [5, 10, 20]
    # 종가MA 대비 종가비율
    for window in windows:
        data['close_ma%d_ratio' % window] = (data['close'] - data['close'].rolling(window).mean()) / data[
            'close'].rolling(window).mean()
    # 거래량MA 대비 거래량 비율
    for window in windows:
        data['volume_ma%d_ratio' % window] = (data['volume'] - data['volume'].rolling(window).mean()) / data[
            'volume'].rolling(window).mean()

    # 볼린저 밴드
    data = bollinger_band(data)

    # MACD(이동평균 수렴확산지수)
    data = cal_MACD(data)

    # RSI(상대강도지수)
    data = cal_RSI(data)

    # stochastic oscillator
    data = cal_stochastic_oscillator(data)

    # log return
    data = cal_log_return(data)

    # MFI
    data = cal_MFI(data)

    data.dropna(inplace=True)
    data.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

    return data


def bollinger_band(data, w = 20, k = 2):
    data = data.copy()
    mbb = data['close'].rolling(w).mean()
    ubb = mbb + k * data['close'].rolling(w).std()
    lbb = mbb - k * data['close'].rolling(w).std()
    data['width'] = ubb - lbb

    # 볼린저 밴드에서 종가 위치
    data['bollinger_band'] = (data['close'] - lbb) / data['width']

    # 볼린저 밴드 넓이 증가량
    data['bollinger_band_ratio'] = np.zeros(len(data))
    data.loc[1:, 'bollinger_band_ratio']= (data['width'][1:].values - data['width'][:-1].values) / data['width'][:-1].values

    data.drop(['width'], axis = 1, inplace = True)

    return data


def cal_MACD(data, num_long=12, num_short=26, num_signal=9):
    data = data.copy()
    ema_long = data['close'].ewm(span=num_long, min_periods=num_long - 1).mean()
    ema_short = data['close'].ewm(span=num_short, min_periods=num_short - 1).mean()
    MACD = ema_long - ema_short
    MACD_signal = MACD.ewm(span=num_signal, min_periods=num_signal - 1).mean()
    data['MACD_diff'] = MACD - MACD_signal

    # MACD cross
    data['MACD_cross'] = pd.Series(np.where(data['MACD_diff'] >= 0, 1, -1), index=data.index)
    # 지난 MACD 대비 MACD 비율
    data['MACD_lastMACD_ratio'] = np.zeros(len(data))
    data.loc[1:, 'MACD_lastMACD_ratio'] = (data['MACD_diff'][1:].values - data['MACD_diff'][:-1].values) / data[
                                                                                                               'MACD_diff'][
                                                                                                           :-1].values

    data.drop('MACD_diff', axis=1, inplace=True)

    return data

def cal_RSI(data, period = 9):
    data = data.copy()

    U = np.where(data['close'].diff(1) > 0, data['close'].diff(1), 0)
    D = np.where(data['close'].diff(1) < 0, data['close'].diff(1) * (-1), 0)

    AU = pd.Series(U, index = data.index).rolling(window=period, min_periods=period).mean()
    AD = pd.Series(D, index = data.index).rolling(window=period, min_periods=period).mean()
    data['RSI'] = AU.div(AD+AU)
    return data

def cal_stochastic_oscillator(data, window = 14):
    data = data.copy()
    ndays_high = data['high'].rolling(window, min_periods=1).max()
    ndays_low = data['low'].rolling(window, min_periods=1).min()
    data['sto_K'] = (data['close'] - ndays_low) / (ndays_high - ndays_low)
    data['sto_D'] = data['sto_K'].rolling(3).mean()
    data['sto_cross'] = pd.Series(np.where(data['sto_K'] > data['sto_D'], 1, -1), index=data.index)
    return data


def cal_OBV(data, n=9):
    data = data.copy()
    OBV = []
    OBV.append(data['volume'][0])
    for i in range(1, len(data)):
        if data['close'][i] > data['close'][i - 1]:
            OBV.append(OBV[-1] + data['volume'][i])
        elif data['close'][i] < data['close'][i - 1]:
            OBV.append(OBV[-1] - data['volume'][i])
        else:
            OBV.append(OBV[-1])
    OBV = pd.Series(OBV, index=data.index)
    data['OBV_ewm'] = OBV.ewm(n).mean()

    # 지난 OBV_ewm 대비 OBV_ewm 비율
    data['OBV_lastOBV_ratio'] = np.zeros(len(data))
    data.loc[1:, 'OBV_lastOBV_ratio'] = (data['OBV_ewm'][1:].values - data['OBV_ewm'][:-1].values) / data['OBV_ewm'][
                                                                                                     :-1].values
    data.drop('OBV_ewm', axis=1, inplace=True)

    return data

def cal_log_return(data):
    data = data.copy()
    data['log_return'] = np.zeros(len(data))
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    return data

def cal_MFI(data, window = 10):
    data = data.copy()
    data['TP'] = (data['high'] + data['low'] + data['close']) / 3
    data['PMF'] = 0
    data['NMF'] = 0
    for i in range(len(data)-1):
        if data['TP'].values[i] < data['TP'].values[i+1]:
            data['PMF'].values[i+1] = data['TP'].values[i+1] * data['volume'].values[i+1]
            data['NMF'].values[i+1] = 0
        else:
            data['NMF'].values[i+1] = data['TP'].values[i+1] * data['volume'].values[i+1]
            data['PMF'].values[i+1] = 0
    data['MFR'] = data['PMF'].rolling(window).sum() / data['NMF'].rolling(window).sum()
    data['MFI'] = 1 - 1 / (1 + data['MFR'])

    data.drop(['TP', 'PMF', 'NMF', 'MFR'], axis=1, inplace=True)

    return data


def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)

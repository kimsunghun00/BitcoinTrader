import pyupbit
import time
import datetime
import pandas as pd
import numpy as np
from IPython.display import clear_output
import torch

import utils

class TradingBot:
    def __init__(self, upbit, model, ticker, interval, window_size, baseline,
                 stoploss = -0.03, lossbreak = False,
                 hour = 6):
        """
        upbit(object) : object of class pyupbit
        ticker(str) : KRX-BTC
        interval(str) : minute1, minute3, minute5...
        baseline(float)
        stoploss(float)
        lossbreak(bool) : Decide whether to break the loop when lost
        ratio(float)
        hour(int) : hours
        window(int)
        
        """
        
        self.upbit = upbit
        self.model = model
        self.ticker = ticker
        
        self.interval = interval
        intervals = ["minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60"]
        if self.interval not in intervals:
            raise Exception(f"{self.interval} is not in inervals")
        
        self.window_size = window_size
        self.baseline = baseline

        self.stoploss = stoploss
        if self.stoploss < -1 or self.stoploss > 1:
            raise Exception("'stoploss' must be between -1 and 1")
        if self.stoploss > 0:
            self.stoploss = -self.stoploss
            
        self.lossbreak = lossbreak
        self.hour = hour
        self.fee = 0.0005
        
        self.dataframe = pd.DataFrame(columns = ['time', 'action', 'earning rate', 'profit and loss'])
        
        
    
    def trade(self, krw = None):
        op_mode = True
        
        tuple_list = []

        t_end = time.time() + self.hour * 3600
        while True:
            clear_output(wait=True)
            
            now = datetime.datetime.now()
            price = pyupbit.get_current_price(self.ticker) # 현재가

            # check hold
            if self.upbit.get_balance(self.ticker) == 0:
                hold = False
                coin_balance = 0.
                avg_buy_price = 0.
                earning_rate = 0.
                profit_loss = 0.
            else:
                hold = True
                coin_balance = self.upbit.get_balance(self.ticker)
                avg_buy_price = self.upbit.get_avg_buy_price(self.ticker) + 0.00001
                earning_rate = (price - avg_buy_price) / avg_buy_price
                estimated_balance = coin_balance * price
                profit_loss = int(estimated_balance * earning_rate  - estimated_balance * self.fee)
            

            ohlcv = pyupbit.get_ohlcv(self.ticker, self.interval, count = 200)
            
            is_bull = self.bull_market(ohlcv, self.ticker)
            if is_bull:
                print('상승장')
            else:
                print('하락장')

            # position
            bb, rsi, (sto_K, sto_D, sto_cross) = self.get_position(ohlcv)
            buy_condition = bb < 0.5 and rsi < 0.3 and sto_K < 0.2 and sto_D < 0.2 and sto_cross == 1
            print('bb : {:.2f}, rsi : {:.2f}'.format(bb * 100, rsi * 100))
            print('%K : {:.2f}, %D : {:.2f}, cross : {}'.format(sto_K * 100, sto_D * 100, sto_cross))
            print()

            # predict
            low_pred, close_pred, high_pred = self.predict(ohlcv)
            print("5분 후 예상 저가 수익률 : {:.3f}%".format(low_pred * 100))
            print("5분 후 예상 종가 수익률 : {:.3f}%".format(close_pred * 100))
            print("5분 후 예상 고가 수익률 : {:.3f}%".format(high_pred * 100))
            print()
    

            # 매초마다 조건을 확인한 후 매수시도
            if op_mode is True and price is not None and hold is False and buy_condition:
                # 매수
                print('매수')
                if krw is not None:
                    krw_balance = krw
                else:
                    krw_balance = self.upbit.get_balance("KRW") * 0.9
                self.upbit.buy_market_order(self.ticker, krw_balance) # 시장가 매수
                print("매수")
                hold = True
                
                # save result
                profit_loss = int(-krw_balance * self.fee)
                tuple_list.append((now, "매수", 0, profit_loss))
    
    
            # 매수 뒤 조건을 확인한 후 매도
            if op_mode is True and hold is True and earning_rate >= self.baseline:
                coin_balance = self.upbit.get_balance(self.ticker)
                self.upbit.sell_market_order(self.ticker, coin_balance)  # 시장가 매도
                print("매도")
                print("{:.2f}%".format(earning_rate * 100))
                hold = False


                # save result
                estimated_balance = coin_balance * price
                profit_loss = int(estimated_balance * earning_rate - estimated_balance * self.fee)
                tuple_list.append((now, "매도", earning_rate, profit_loss))
                print(profit_loss, '원')
                
            # 손절매
            if op_mode is True and hold is True and earning_rate < self.stoploss:
                coin_balance = self.upbit.get_balance(self.ticker)
                self.upbit.sell_market_order(self.ticker, coin_balance)
                print("손절")
                print("{:.2f}%".format(earning_rate * 100))
                hold = False
                
                # save result
                estimated_balance = coin_balance * price
                profit_loss = int(estimated_balance * earning_rate  - estimated_balance * self.fee)
                tuple_list.append((now, "손절", earning_rate, profit_loss))
                print(profit_loss, '원')
                
                time.sleep(10)
    
            print("현재시간: {}, 현재가: {}, 평균단가: {:.2f}, 수익률: {:.3f}%, 손익: {}원".format(now.strftime('%Y-%m-%d %H:%M:%S'),
                                                                                  price, avg_buy_price, earning_rate * 100, profit_loss))
            print(f"보유상태: {hold}, 동작상태: {op_mode}")
            print()
            time.sleep(1)

            if time.time() > t_end and hold is False:
                break
        
        for t, value, earning_rate, profit in tuple_list:
            self.dataframe = self.dataframe.append({'time':t.strftime('%Y-%m-%d %H:%M:%S'), 'action':value,
                                                    'earning rate':'{:.2f}%'.format(earning_rate * 100), 'profit and loss':profit}, 
                                                    ignore_index = True)
        print('종료')

            
    def get_trading_data(self):
        trading_result = self.dataframe.copy()
        trading_result['time'] = pd.to_datetime(trading_result['time'])
        trading_result.set_index('time', inplace = True)
        
        return trading_result
        

    def bull_market(self, data, ticker):
        ema = data['close'].ewm(5).mean()
        price = pyupbit.get_current_price(ticker)
        last_ema = ema[-2]

        if price > last_ema:
            return True
        else:
            return False

    def get_position(self, data):
        bb = utils.bollinger_band(data)['bollinger_band'][-1]

        rsi = utils.cal_RSI(data)['RSI'][-1]

        sto = utils.cal_stochastic_oscillator(data)[['sto_K', 'sto_D', 'sto_cross']]
        sto_K, sto_D, sto_cross = sto.iloc[-1, 0], sto.iloc[-1, 1], sto.iloc[-1, 2]

        return bb, rsi, (sto_K, sto_D, sto_cross)
        
    def predict(self, data):
        # data processing
        data = utils.add_variables(data)
        #data = data.tail(self.window_size)

        feature_list = []
        for i in range(len(data) - self.window_size):
            feature_list.append(np.array(data.iloc[i: i + self.window_size]))
        data = np.array(feature_list)
        
        # predict
        data = torch.tensor(data[-1], dtype = torch.float)
        data = data.view(1, self.window_size, data.shape[1])
        prediction = self.model(data).detach().numpy()
        low_pred, close_pred, high_pred = prediction[0,0], prediction[0,1], prediction[0,2]

        return low_pred, close_pred, high_pred
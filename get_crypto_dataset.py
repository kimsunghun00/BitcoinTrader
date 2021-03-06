import pyupbit
import tqdm
import time
import datetime
import argparse
import pandas as pd

from utils import add_variables

class MakeDataset:
    def __init__(self, ticker, interval, frm, to = None):
        self.ticker = ticker
        self.interval = interval
        self.frm = frm
        self.to = to
        
        self.dataset = None
        
    def get_dataset(self):
        self.dataset = self.get_ohlcv_continue(self.ticker, self.interval, self.frm, self.to)

        print('preprocessing..')
        self.dataset = self.preprocess(self.dataset)

        print('add variables..')
        self.dataset = add_variables(self.dataset)

        print('done!')
        
        return self.dataset
    
    def get_ohlcv_continue(self, ticker, interval, frm, to = None):
    
        """
    
        ticker(str) : KRW-BTC
        interval : minute1, minute3, minute5, minute10, minute15, minute60
        frm(str) :
    
        """
    
        if isinstance(frm, str):
            frm = pd.to_datetime(frm).to_pydatetime()
        
        if to is not None:
            if isinstance(to, str):
                to = pd.to_datetime(to).to_pydatetime()
        else:
            to = datetime.datetime.now().replace(microsecond=0)
    
    
        if interval == "minute1":
            count = 60
            date_list = list(pd.date_range(start = frm, end = to, freq = 'H').to_pydatetime())
        elif interval == "minute3":
            count = 60
            date_list = list(pd.date_range(start = frm, end = to, freq = '3H').to_pydatetime())
        elif interval == "minute5":
            count = 60
            date_list = list(pd.date_range(start = frm, end = to, freq = '5H').to_pydatetime())
        elif interval == "minute10":
            count = 144
            date_list = list(pd.date_range(start = frm, end = to, freq = 'D').to_pydatetime())
        elif interval == "minute15":
            count = 96
            date_list = list(pd.date_range(start = frm, end = to, freq = 'D').to_pydatetime())
        elif interval == "minute30":
            count = 48
            date_list = list(pd.date_range(start = frm, end = to, freq = 'D').to_pydatetime())
        elif interval == "minute60":
            count = 24
            date_list = list(pd.date_range(start = frm, end = to, freq = 'D').to_pydatetime())

        dataframes = []
        for date in tqdm.tqdm(date_list[1:]):
            try:
                df = pyupbit.get_ohlcv(ticker, interval, count = count, to = date)
                dataframes.append(df)
                time.sleep(0.1)
            except:
                pass
        
        data = pd.concat(dataframes)
        # ?????? ????????? ??????
        data = data.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
        data.sort_index(inplace=True)

        return data
    
    def preprocess(self, data, window=2):
        data = data.copy()
        data['low_pred'] = data['low'].rolling(window).apply(lambda x: (x.iloc[window-1] - x.iloc[0]) / x.iloc[0])
        data['low_pred'] = data['low_pred'].shift(1)
        data['close_pred'] = data['close'].rolling(window).apply(lambda x: (x.iloc[window - 1] - x.iloc[0]) / x.iloc[0])
        data['close_pred'] = data['close_pred'].shift(1)
        data['high_pred'] = data['high'].rolling(window).apply(lambda x: (x.iloc[window - 1] - x.iloc[0]) / x.iloc[0])
        data['high_pred'] = data['high_pred'].shift(1)

        data.dropna(inplace = True)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type = str, default = 'KRW-BTC')
    parser.add_argument('--interval', type = str, default = 'minute1')
    parser.add_argument('--frm', type = str, default = '2021-01-01 00:00:00')
    parser.add_argument('--to', type = str, default = None)
    parser.add_argument('--window', type=int, default=5)
    args = parser.parse_args()

    print(f'{args.ticker} data is being obtained and processed')
    mk = MakeDataset(args.ticker, args.interval, frm = args.frm, to = args.to)
    data = mk.get_dataset()
    data.to_csv("./data/{}.csv".format(args.ticker))
    print('save completed')

import os
import pandas as pd
import numpy as np
import tqdm
import torch

class Dataset:
    def __init__(self, file_path, window_size = 20, test_size = 10000):
        
        self.file_path = file_path

        self.tickers = []
        for file in os.listdir(self.file_path):
            ticker = file.split('.')[0]
            self.tickers.append(ticker)
        
        self.window_size = window_size

        self.train_dataframe_list = []
        for ticker in tqdm.tqdm(self.tickers):
            if ticker == 'KRW-BTC':
                train_df = self.read_file(ticker)[:-test_size]
                self.test_dataframe = self.read_file(ticker)[-test_size:]
                self.train_dataframe_list.append(train_df)
            else:
                df = self.read_file(ticker)
                self.train_dataframe_list.append(df)

        self.num_labels = 3
        self.num_features = len(self.train_dataframe_list[0].columns) - self.num_labels
        self.feature_cols = self.train_dataframe_list[0].columns[self.num_labels:].to_list()

    def get_dataset(self):
        train_features_numpy_list = []
        train_labels_numpy_list = []
        # Train Dataset
        for i, train_dataframe in enumerate(tqdm.tqdm(self.train_dataframe_list)):
            train_feature = train_dataframe[self.feature_cols]
            train_label = train_dataframe[['low_pred', 'close_pred', 'high_pred']]

            # Normalize
            train_feature = train_feature.clip(-5, 5)
            # make dataset
            train_features_numpy, train_labels_numpy = self.make_dataset(train_feature, train_label, window_size = self.window_size)
            train_features_numpy_list.append(train_features_numpy)
            train_labels_numpy_list.append(train_labels_numpy)

        train_features = np.concatenate(train_features_numpy_list, axis = 0)
        train_labels = np.concatenate(train_labels_numpy_list)
        print(self.tickers, 'are combined as training set')

        # to tensor
        train_features = torch.tensor(train_features, dtype = torch.float)
        train_labels = torch.tensor(train_labels, dtype = torch.float)


        # Test Dataset
        test_feature = self.test_dataframe[self.feature_cols]
        test_label = self.test_dataframe[['low_pred', 'close_pred', 'high_pred']]
        # Normalize
        test_feature = test_feature.clip(-5, 5)
        # make dataset
        test_features_numpy, test_labels_numpy = self.make_dataset(test_feature, test_label, window_size=self.window_size)
        # to tensor
        test_features = torch.tensor(test_features_numpy, dtype=torch.float)
        test_labels = torch.tensor(test_labels_numpy, dtype=torch.float)

        print('completed')
        return (train_features, train_labels), (test_features, test_labels)

    def get_test_dataset(self):
        test_feature = self.test_dataframe[self.feature_cols]
        test_label = self.test_dataframe[['low_pred', 'close_pred', 'high_pred']]
        # Normalize
        test_feature = test_feature.clip(-5, 5)
        # make dataset
        test_features_numpy, test_labels_numpy = self.make_dataset(test_feature, test_label,
                                                                   window_size=self.window_size)
        # to tensor
        test_features = torch.tensor(test_features_numpy, dtype=torch.float)
        test_labels = torch.tensor(test_labels_numpy, dtype=torch.float)

        return test_features, test_labels

    def get_dataframes(self):
        return self.train_dataframe_list
          
    def make_dataset(self, data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i : i+window_size]))
            label_list.append(np.array(label.iloc[i + window_size]))
        return np.array(feature_list), np.array(label_list)

    def read_file(self, ticker):
        data = pd.read_csv(os.path.join(self.file_path, f'{ticker}.csv'), index_col = 0)
        data.index = pd.to_datetime(data.index)
        return data
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class IBMDataset:

    def ibm_credit_card(self):
        cards = pd.read_csv("data/ibm_credit_card/sd254_cards.csv")
        users = pd.read_csv("data/ibm_credit_card/sd254_users.csv")
        _transactions = pd.read_csv("data/ibm_credit_card/credit_card_transactions-ibm_v2.csv")
        # user_transactions = pd.read_csv("../data/ibm_credit_card/User0_credit_card_transactions.csv")

        # merge the dataframes
        result_df = _transactions.merge(users, left_on='User', right_index=True, how='left')
        transactions = result_df.merge(cards, left_on=['User', 'Card'], right_on=['User', 'CARD INDEX'], how='left')

        # Combine the year, month, day, and time columns into a single datetime column
        transactions['datetime'] = pd.to_datetime(transactions[['Year', 'Month', 'Day']].astype(str)
                                                    .apply('-'.join, 1) + ' ' + transactions['Time'])

        # Convert the datetime column to Unix timestamp (in seconds)
        transactions['unix_timestamp'] = (transactions['datetime'] - datetime(1970,1,1)).dt.total_seconds().astype(int)

        # Drop unnecessary columns
        transactions.drop(columns=['Card', 'Month', 'Day', 'Time'], inplace=True)

        # Create the resulting dataframe
        df = pd.DataFrame({
            'customer.id': transactions.User,
            'card.id': transactions['Card Number'],
            'amount_signed': [float(x[1:]) for x in transactions.Amount],
            'timestamp': transactions.unix_timestamp,
            'date': transactions.datetime,
            'merchant.name': transactions['Merchant Name'],
            'merchant.city': transactions['Merchant City'],
            'merchant.state': transactions['Merchant State'],
            # 'card.mcc': transactions['MCC'],
            'is_fraud': transactions["Is Fraud?"],
            'age': transactions['Current Age'],
            'chip': transactions['Use Chip'],
            'gender': transactions['Gender'],
            'customer.city': transactions['City'],
            'customer.state': transactions['State'],
            'score': transactions['FICO Score'],
            'num_cards': transactions['Num Credit Cards'],
            'total_debt': [float(x[1:]) for x in transactions['Total Debt']],
            'credit_limit': [float(x[1:]) for x in transactions['Credit Limit']],
            'card.brand': transactions['Card Brand'],
            'latitude': transactions['Latitude'],
            'longitude': transactions['Longitude']
        })

        # Process new columns
        df['direction'] = df['amount_signed'].apply(lambda x: 'inbound' if x < 0 else 'outbound')
        df['amount_usd'] = np.abs(np.array(df['amount_signed']))
        df['log_amount'] = np.log(1 + np.array(df.amount_usd))

        return df
    
    def process_dataset(self):
        print('Loading the data...')
        df = self.ibm_credit_card()
        print('Data loaded.')

        print('Processing data with new columns...')
        gdf = df.groupby('customer.id')

        new_groups = []
        for _, group in tqdm(gdf):
            # figure out known merchants
            known_merchants = group['merchant.name'].duplicated().astype(int)
            
            # create a copy of the group
            g_new = group.copy()
            g_new['is_known_merchant'] = known_merchants
            
            # time difference for the customer
            g_new['log_timediff'] = np.log(1 + g_new['date'].diff().dt.seconds).fillna(0)
            
            # customer address and merchant address
            g_new['same_city'] = (group['merchant.city'] == group['customer.city'])
            g_new['same_state'] = (group['merchant.state'] == group['customer.state'])
            
            new_groups.append(g_new)

        data = pd.concat(objs=new_groups)

        print('Data processed and concatenated.')
        
        return data

def convert_time_features(df, datetime_column_name = 'datetime'):
    """
	Converts `datetime_column_name` from DataFrame to 2D sin/cos encoded
	numerical features. Encodes
	- day of the week
	- day of the month
	- month of the year
    - hour of day
    - minute of hour

	:returns
	- np.array of shape (nrows, 10)
	"""

    # Convert the date column to datetime objects
    dates = pd.to_datetime(df[datetime_column_name])

    # Calculate the angles for each time feature
    day_of_week_angle = 2 * np.pi * dates.dt.dayofweek / 7
    day_of_month_angle = 2 * np.pi * (dates.dt.day - 1) / (dates.dt.daysinmonth - 1)
    month_angle = 2 * np.pi * (dates.dt.month - 1) / 12
    hour_angle = 2 * np.pi * (dates.dt.hour - 1) / 24
    minute_angle = 2 * np.pi * (dates.dt.minute - 1) / 60

    # Transform the angles into continuous features using sine and cosine functions
    day_of_week = np.column_stack((np.sin(day_of_week_angle), np.cos(day_of_week_angle)))
    day_of_month = np.column_stack((np.sin(day_of_month_angle), np.cos(day_of_month_angle)))
    month = np.column_stack((np.sin(month_angle), np.cos(month_angle)))
    hour = np.column_stack((np.sin(hour_angle), np.cos(hour_angle)))
    minute = np.column_stack((np.sin(minute_angle), np.cos(minute_angle)))

    # Combine the continuous features into a single numpy array
    time_features = np.hstack((day_of_week, day_of_month, month, hour, minute))

    return time_features

class TransactionDataset(Dataset):
    def __init__(self, gdf, oh_c, oh_m, seq_length):
        self.gdf = gdf
        self.len = len(gdf)
        self.groups = list(gdf.groups)
        self.oh_customer_state = oh_c
        self.oh_merchant_state = oh_m
        self.seq_length = seq_length

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.get_group(idx)
        data, label = self.process_rows(g)
        return data, label
    
    def get_group_by_label(self, group_ix, transaction_ix):
        g = self.gdf.get_group(group_ix)

        loc_idx = g.index.get_loc(transaction_ix)

        start_ix = loc_idx - self.seq_length + 1
        if start_ix < 0:
            return None
        
        g = g.iloc[start_ix:loc_idx+1]
        return self.process_rows(g)
        
    
    def get_group(self, idx):
        # get the group
        g = self.gdf.get_group(self.groups[idx])

        # get the sequence length of customers transactions
        l = len(g)
        start_ix = random.sample(range(l - self.seq_length), 1)[0] + 1
        g = g.iloc[start_ix:start_ix + self.seq_length]
        return g
    
    def get_group_all_sequences(self, idx):
        # get the group
        g = self.gdf.get_group(self.groups[idx])

        # process values and labels
        processed, _ = self.process_rows(g)
        labels = np.array(g['is_fraud'] == 'Yes', dtype=float)

        max_len = processed.shape[0]
        sequences = [processed[start_ix:start_ix+self.seq_length] for start_ix in range(max_len - self.seq_length + 1)]
        sequences = np.stack(sequences, axis=0)
        labels = np.hstack([labels[start_ix + self.seq_length - 1] for start_ix in range(max_len - self.seq_length + 1)])
        
        return sequences, labels

    
    def process_rows(self, g):
        # process the data
        direction = np.array(g['direction'] == 'inbound', dtype=float)
        brand = np.array(g['card.brand'] == 'Visa', dtype=float)
        gender = np.array(g['gender'] == 'female', dtype=float)
        same_city = np.array(g['same_city'], dtype=float)
        same_state = np.array(g['same_state'], dtype=float)
        known_merchant = np.array(g['is_known_merchant'], dtype=float)
        debt = np.array(np.log(1 + g['total_debt']), dtype=float)
        limit = np.array(np.log(1 + g['credit_limit']), dtype=float)
        
        customer_state = self.oh_customer_state.transform(np.array(g['customer.state']).reshape(-1,1))
        merchant_state = self.oh_merchant_state.transform(np.array(g['merchant.state']).reshape(-1,1))

        Fnum = np.array(g[['age', 'num_cards', 'log_amount', 'log_timediff', 'latitude', 'longitude']], dtype=float)
        if len(Fnum.shape) == 1:
            Fnum = Fnum.reshape(-1,1).transpose()
            
        Fcat = np.vstack((debt, limit, direction, brand, gender, same_city, same_state, known_merchant)).transpose()
        Fdatetime = convert_time_features(g, 'date')

        F = np.hstack((customer_state, merchant_state, Fnum, Fcat, Fdatetime))

        # get the label of the last transaction in the sequence
        label = np.array(g.iloc[-1]['is_fraud'] == 'Yes', dtype=float)

        return np.array(F, dtype=float), label

### Functions ###

def fit_onehot_encoders(data):
    oh_customer_state = OneHotEncoder(min_frequency=1, sparse_output=False, handle_unknown='infrequent_if_exist')
    oh_merchant_state = OneHotEncoder(min_frequency=5, sparse_output=False, handle_unknown='infrequent_if_exist')
    
    oh_customer_state.fit(np.array(data['customer.state']).reshape(-1,1))
    oh_merchant_state.fit(np.array(data['merchant.state']).reshape(-1,1))
    
    return oh_customer_state, oh_merchant_state

def train_val_test_indexes(data):
    # Split indexes by customers
    grouped_df = data.groupby('customer.id')
    groups = np.array(list(grouped_df.groups))

    index_list = list(range(len(groups)))
    tmp, _test_ix = train_test_split(index_list, test_size=0.2, random_state=31)
    _train_ix, _val_ix = train_test_split(tmp, test_size=0.3, random_state=31)

    print(len(_train_ix))
    print(len(_val_ix))
    print(len(_test_ix))

    train_ix = []
    val_ix = []
    test_ix = []

    for gix in groups[_train_ix]:
        g = grouped_df.get_group(gix)
        indexes = g.index
        train_ix.extend(indexes)

    for gix in groups[_val_ix]:
        g = grouped_df.get_group(gix)
        indexes = g.index
        val_ix.extend(indexes)

    for gix in groups[_test_ix]:
        g = grouped_df.get_group(gix)
        indexes = g.index
        test_ix.extend(indexes)

    return train_ix, val_ix, test_ix

def train_val_test_datasets(data, train_ix, val_ix, test_ix, oh_customer_state, oh_merchant_state):
    train_data = data.loc[train_ix]
    val_data = data.loc[val_ix]
    test_data = data.loc[test_ix]

    train_dataset = TransactionDataset(train_data.groupby('customer.id'), oh_customer_state, oh_merchant_state, 10)
    val_dataset = TransactionDataset(val_data.groupby('customer.id'), oh_customer_state, oh_merchant_state, 10)
    test_dataset = TransactionDataset(test_data.groupby('customer.id'), oh_customer_state, oh_merchant_state, 10)

    return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
    
class CustomUpsampleDataset(Dataset):
    def __init__(self, dataset, pidx, nidx, length, threshold=0.5):
        self.dataset = dataset
        self.positive_idx = pidx
        self.negative_idx = nidx
        self.length = length
        self.threshold = threshold

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        while True:
            if random.random() > self.threshold:
                idx = random.sample(list(self.positive_idx.keys()), 1)[0]
                i = random.sample(list(self.positive_idx[idx]), 1)[0]
                t = self.dataset.get_group_by_label(idx, i)
            else:
                idx = random.sample(list(self.negative_idx.keys()), 1)[0]
                i = random.sample(list(self.negative_idx[idx]), 1)[0]
                t = self.dataset.get_group_by_label(idx, i)
            
            if t is not None:
                x, y = t
        
                x = torch.tensor(x, dtype=torch.float32).squeeze(0)
                y = torch.tensor(y, dtype=torch.float32)
                return x, y

    @property    
    def len(self):
        return len(self.dataset)
    

def positive_negative_indexes(train_data):
    pos_idx = {}
    neg_idx = {}
    for cid, group in train_data.groupby('customer.id'):
        p = group[group['is_fraud'] == 'Yes'].index
        n = group[group['is_fraud'] == 'No'].index

        if len(p) > 0:
            pos_idx[cid] = p
        if len(n) > 0:
            neg_idx[cid] = n

    return pos_idx, neg_idx


import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from datetime import datetime
import copy
import cloudpickle

def plotsize(x,y):
    sns.set(rc={'figure.figsize':(x,y)})
    
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import BCELoss, NLLLoss, MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#########################
### utility functions ###
#########################

def moving_average(values, window_size=30):
    cum_sum = [0]
    moving_averages = []
    for i, x in enumerate(values, 1):
        cum_sum.append(cum_sum[-1] + x)
        if i >= window_size:
            moving_avg = (cum_sum[-1] - cum_sum[-1 - window_size]) / window_size
            moving_averages.append(moving_avg)
        else:
            moving_averages.append(cum_sum[-1] / i)

    return np.array(moving_averages)

###########################################################################
############################ Utility functions ############################
###########################################################################

df = pd.read_csv('../data/czech_banking_dataset.csv', low_memory=False)

df['datetime'] = pd.to_datetime(df.date)
df = df[['trans_id', 'direction', 'type', 'client_id', 'amount', 'datetime']]
df['timestamp'] = df['datetime'].astype('datetime64[ns]').astype('int64') // 10**9

# create the sine and cosine versions of the datetime features
df['day_of_week_angle'] = 2 * np.pi * df['datetime'].dt.dayofweek / 7
df['day_of_month_angle'] = 2 * np.pi * (df['datetime'].dt.day - 1) / (df['datetime'].dt.daysinmonth - 1)
df['month_angle'] = 2 * np.pi * (df['datetime'].dt.month - 1) / 12
df['year_angle'] = 2 * np.pi * (df['datetime'].dt.year - df['datetime'].dt.year.min()) / (df['datetime'].dt.year.max() - df['datetime'].dt.year.min())

# Transform the angles into continuous features using sine and cosine functions
df['day_of_week_sin'] = np.sin(df['day_of_week_angle'])
df['day_of_week_cos'] = np.cos(df['day_of_week_angle'])
df['day_of_month_sin'] = np.sin(df['day_of_month_angle'])
df['day_of_month_cos'] = np.cos(df['day_of_month_angle'])
df['month_sin'] = np.sin(df['month_angle'])
df['month_cos'] = np.cos(df['month_angle'])
df['year_sin'] = np.sin(df['year_angle'])
df['year_cos'] = np.cos(df['year_angle'])

# drop the unnecessary columns
df = df.drop(columns=['day_of_week_angle', 'day_of_month_angle', 'month_angle', 'year_angle'])

_df = df.rename(columns={'trans_id': 'id', 'client_id': 'customer.id', 'type': 'scheme'})
df = _df.groupby('customer.id').filter(lambda x: len(x) > 200)

grouped_df = df.groupby('customer.id')

####################################################################
############################ DataLoader ############################
####################################################################

def convert_time_features(df):
    # Convert the date column to datetime objects
    dates = pd.to_datetime(df['datetime'])

    # Calculate the angles for each time feature
    day_of_week_angle = 2 * np.pi * dates.dt.dayofweek / 7
    day_of_month_angle = 2 * np.pi * (dates.dt.day - 1) / (dates.dt.daysinmonth - 1)
    month_angle = 2 * np.pi * (dates.dt.month - 1) / 12

    # Transform the angles into continuous features using sine and cosine functions
    day_of_week = np.column_stack((np.sin(day_of_week_angle), np.cos(day_of_week_angle)))
    day_of_month = np.column_stack((np.sin(day_of_month_angle), np.cos(day_of_month_angle)))
    month = np.column_stack((np.sin(month_angle), np.cos(month_angle)))

    # Combine the continuous features into a single numpy array
    time_features = np.hstack((day_of_week, day_of_month, month))

    return time_features

def fit_onehot_encoder(grouped_df):
    encoder = OneHotEncoder(sparse_output=False)
    v = np.array(grouped_df.obj.scheme).reshape(-1, 1)
    encoder.fit(v)
    return encoder

onehot_encoder = fit_onehot_encoder(grouped_df)
MX_length = 500

class TransactionDataset(Dataset):
    
    def __init__(self, grouped_df, onehot_encoder, ids=None, *, min_length=50, max_length=500):
        self.ids = ids if ids is not None else list(grouped_df.groups)
        self.gdf = grouped_df
        self.encoder = onehot_encoder
        self.min_length = min_length
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            if key.step is not None:
                return [self._get_item(i) for i in range(key.start, key.stop, key.step)]
            else:
                return [self._get_item(i) for i in range(key.start, key.stop)]
        elif isinstance(key, int):
            return self._get_item(key)
        
    def _get_item(self, idx):
        x = self._get_array(idx)
        # return self.get_fixed_history(x)
        return self.get_variable_history(x)
        
    def get_fixed_history(self, transaction_sequence):
        start_ix = random.randint(0, transaction_sequence.shape[0] - self.min_length - 1)
        
        return (
            transaction_sequence[start_ix:start_ix + self.min_length],
            transaction_sequence[start_ix+1:start_ix + self.min_length+1]
        )

    def get_variable_history(self, transaction_sequence):
        max_len = min(transaction_sequence.shape[0], self.max_length)
        seq_length = random.randint(self.min_length, max_len - 1)
        start_ix = random.randint(0, max_len - seq_length - 1)
        
        return (
            transaction_sequence[start_ix:start_ix + seq_length],
            transaction_sequence[start_ix+1:start_ix + seq_length+1]
        )
    
    def get_consequent_history_of_customer(self, idx: int, seq_length: int, max_history_length=30):
        transaction_sequence = self._get_array(idx)
        max_length = min(transaction_sequence.shape[0] - seq_length, max_history_length)
        
        histories_input = []
        histories_target = []
        
        for start_ix in range(max_length):
            input_seq = transaction_sequence[start_ix:start_ix + seq_length]
            target_seq = transaction_sequence[start_ix+1:start_ix + seq_length+1]
            histories_input.append(input_seq)
            histories_target.append(target_seq)
            
        return histories_input, histories_target
        
    def _get_array(self, idx):
        g = self.get_group(idx)
        t = self.get_time_features(g)         # continuous       (seq_length, 6)
        a = self.get_amount(g)                # continuous       (seq_length, 1)
        binary, onehot = self.get_onehot(g)   # binary / onehot  (seq_length, 1) and (seq_length, 5)
                                              # total:           13
        
        continuous = np.concatenate((t, a), axis=1)
        continuous = torch.from_numpy(np.array(continuous, dtype=np.float32))
        binary = torch.from_numpy(np.array(binary, dtype=np.float32))
        onehot = torch.from_numpy(np.array(onehot, dtype=np.float32))
        
        train = (continuous, binary, onehot)
        return torch.cat(train, dim=-1)
    
    def get_customer_len(self, idx):
        g = self.get_group(idx)
        return len(g)
    
    def get_group(self, ix):
        return self.gdf.get_group(self.ids[ix])
    
    def get_time_features(self, g):
        """Returns an array of shape (6, seq_length) with sin/cos repr. of weekday, monthday, month."""
        return convert_time_features(g)
    
    def get_amount(self, g):
        """Returns the logarithmized amount."""
        return np.log(1 + np.array(g.amount)).reshape(-1,1)
    
    def get_onehot(self, g):
        direction = g.direction == 'inbound'
        binary = np.array(direction, dtype=float)
        
        onehot = self.encoder.transform(np.array(g['scheme']).reshape(-1, 1))
        return binary.reshape(-1,1), onehot
    
    @property
    def continuous_length(self):
        x = self.get_time_features(self.get_group(1))
        y = self.get_amount(self.get_group(1))
        return x.shape[-1] + y.shape[-1]
    
    @property
    def binary_length(self):
        x = self.get_onehot(self.get_group(1))[0]
        return x.shape[-1]
    
    @property
    def onehot_length(self):
        x = self.get_onehot(self.get_group(1))[1]
        return x.shape[-1]
    
    @property
    def feature_length(self):
        x = self._get_array(1)
        return x.shape[1]
    
    @property
    def type_lengths(self):
        return [self.continuous_length, self.binary_length, self.onehot_length]
    
    def print_lengths(self):
        print('Continuous length:', self.continuous_length)
        print('Binary length:    ', self.binary_length)
        print('Onehot length:    ', self.onehot_length)
        print('Total feature length:', self.feature_length)

##########################################################################
############################ Model definition ############################
##########################################################################

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, dataset, num_features, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        m = self.generate_square_subsequent_mask()
        self.mask = m
        
        self.transformer_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layers, num_layers)

        self.mean = nn.Linear(d_model, dataset.continuous_length)
        self.binary_model = nn.Linear(d_model, dataset.binary_length)
        self.onehot_model = nn.Linear(d_model, dataset.onehot_length)
        
    def generate_square_subsequent_mask(self, size=200): # Generate mask covering the top right triangle of a matrix
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1).to(device)
        return mask.to(device)
        
    def forward(self, src, padding_mask=None, causality_mask=None):
    
        # process through the model
        src = self.embedding(src)
        src = self.positional_encoding(src)
        mask_size = src.shape[1]
        # print(mask_size)
        m = self.generate_square_subsequent_mask(mask_size)
        x = self.transformer(
            src, src,                                                                 # target and memory are the same
            tgt_mask=m, memory_mask=m,                                                # triangular masks so that we do not attend to the future tokens
            tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask,  # padding mask, so that we are not training on padded parts of the sequences
        )
        
        # process the outpus
        c_mean = self.mean(x)
        b = torch.sigmoid(self.binary_model(x))
        oh = self.onehot_model(x) # should be raw logits
        
        return c_mean, b, oh
    
####################################################################################
############################ Load model and other stuff ############################
####################################################################################

newest_file = sorted(os.listdir('saved_models'), key=lambda x: os.path.getmtime(os.path.join('saved_models', x)))[-1]
print(f'Loading model {newest_file}.')

with open(f'saved_models/{newest_file}', 'rb') as f:
    saving = cloudpickle.load(f)

model = saving['model']
optimizer = saving['optimizer']
scheduler = saving['scheduler']
ids = saving['ids']
train_ix = saving['train_ix']
test_ix = saving['test_ix']

# dataset
dataset = TransactionDataset(grouped_df, onehot_encoder, ids, min_length=30, max_length=250)

# minibatching
def create_minibatch(dataset, ixs, batch_size, n_batches, max_sequence_length=200):
    
    for _ in range(n_batches):
        # Randomly sample starting indices for the sequences
        start_indices = random.sample(ixs, batch_size)

        # Initialize input and target tensors with zero-padding
        inputs = torch.zeros(batch_size, max_sequence_length, dataset.feature_length)
        targets = torch.zeros(batch_size, max_sequence_length, dataset.feature_length)
        # padding_masks = torch.ones(batch_size, max_sequence_length, dtype=bool)
        padding_masks = torch.ones(batch_size, max_sequence_length, dtype=torch.float32)

        # Fill input and target tensors with data from the dataset
        for i, start_idx in enumerate(start_indices):
            input_data, target_data = dataset[start_idx]
            
            seq_length = input_data.size(0)
            inputs[i, :seq_length, :] = input_data
            targets[i, :seq_length, :] = target_data

            # Fill the masks with 1 for non-padded elements
            # padding_masks[i, :seq_length] = False
            padding_masks[i, :seq_length] = 0
        
        # Create target attention mask
        # causality_mask = torch.triu(torch.ones(max_sequence_length, max_sequence_length), diagonal=1).bool()
        
        # yield inputs, targets, input_masks, target_masks
        yield inputs, targets, padding_masks#, causality_mask

# hyperparameters and device
input_dim = dataset.feature_length
d_model = 512 # 512
nhead = 8               # 8
num_layers = 6  # 6
dim_feedforward = 2048   # 2048
dropout = 0.1
device = torch.device('mps')
# device = torch.device('cpu')
# device = torch.device('gpu')

# parameters
label_smoothing = 0.1
type_lengths = dataset.type_lengths

########################################################################
############################ Loss functions ############################
########################################################################

def gaussian_log_likelihood(y_pred, log_var, y_true, mask):
    """
    Computes the Gaussian log-likelihood loss with padding mask.

    Parameters:
    y_pred (torch.Tensor): Predicted values, typically the mean of the Gaussian distribution.
    y_true (torch.Tensor): Ground truth values.
    log_var (torch.Tensor): Logarithm of the variance of the Gaussian distribution.
    mask (torch.Tensor): Padding mask with the same shape as y_true and y_pred.

    Returns:
    torch.Tensor: Gaussian log-likelihood loss.
    """
    mse = F.mse_loss(y_pred, y_true, reduction='none')
    
    # Apply the mask to the mse and log_var tensors
    m = mask.unsqueeze(-1).expand_as(y_pred)
    masked_mse = mse * m
    masked_log_var = log_var * m

    loss = 0.5 * (masked_log_var + masked_mse / torch.exp(masked_log_var))
    
    # Calculate the mean loss considering only the non-padded elements
    loss = loss.sum() / m.sum()
    return loss

def mse_loss(y_pred, y_true, mask):
    """
    Computes the MSE loss with padding mask.

    Parameters:
    y_pred (torch.Tensor): Predicted values.
    y_true (torch.Tensor): Ground truth values.
    mask (torch.Tensor): Padding mask with the same shape as y_true and y_pred.

    Returns:
    torch.Tensor: Masked MSE loss.
    """
    mse = F.mse_loss(y_pred, y_true, reduction='none')
    
    # Apply the mask to the mse and log_var tensors
    m = mask.unsqueeze(-1).expand_as(y_pred)
    masked_mse = mse * m
    
    # Calculate the mean loss considering only the non-padded elements
    loss = masked_mse.sum() / m.sum()
    return loss

def binary_loss(y_pred, y_true, mask, label_smoothing=0.1):
    m = mask.unsqueeze(-1).expand_as(y_pred)
    y_pred_masked = y_pred * m
    y_true_masked = (y_true * (1 - label_smoothing * 2) + label_smoothing) * m
    
    binary_crossentropy = nn.BCELoss(reduction='sum')(y_pred_masked, y_true_masked) / m.sum()
    return binary_crossentropy

ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

def onehot_loss(y_pred, y_true, mask, label_smoothing=0.1):
    onehot_indices = torch.argmax(y_true, dim=-1) * mask.long()
    onehot_crossentropy = (ce_loss(y_pred.permute(0,2,1), onehot_indices) * mask).sum() / mask.sum()
    return onehot_crossentropy

def compute_masked_losses(mean, binary, onehot, target, mask):
    c, b, o = target.split(dataset.type_lengths, dim=-1)
    m_loss = mse_loss(mean, c, mask)
    b_loss = binary_loss(binary, b, mask)
    oh_loss = onehot_loss(onehot, o, mask)
    
    return m_loss + b_loss + oh_loss


#######################################################################
############################ Training loop ############################
#######################################################################

# Set the number of epochs, batches per epoch, and batch size
batch_losses = []
num_epochs = 1
n_batches = 30
batch_size = 32
print_every = 10

# Loop over the total number of epochs
for epoch in range(num_epochs):
    model.train()
    # Initialize the minibatch generator
    minibatch_generator = create_minibatch(dataset, train_ix, batch_size, n_batches, max_sequence_length=250)

    # Initialize the epoch loss
    epoch_loss = 0.0

    # Loop over the batches in the minibatch generator
    for i, (inputs, targets, padding_mask) in enumerate(minibatch_generator):
        # Move inputs and targets to the appropriate device
        inputs = inputs.to(device)
        targets = targets.to(device)
        padding_mask = padding_mask.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        # c_mean, c_var, b, o = model(inputs, padding_mask)
        c_mean, b, o = model(inputs, padding_mask)

        # Compute the losses
        # loss = compute_masked_losses(c_mean, c_var, b, o, targets, 1 - padding_mask)
        loss = compute_masked_losses(c_mean, b, o, targets, 1 - padding_mask)

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the epoch loss
        epoch_loss += loss.item()
        batch_losses.append(epoch_loss / (i+1))

        # Print the loss after every `print_every` batches
        if (i + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{n_batches}], Loss: {epoch_loss / (i+1):.4f}")

    # Adjust the learning rate using the scheduler
    scheduler.step()

    # Print the epoch loss
    print(f'Last learning rate: {scheduler.get_last_lr()}')


print('This is finished, nice job.')
print('Saving model.')

model_id = str(datetime.now().strftime('%Y_%m_%d_%H-%M'))

saving = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'ids': ids,
    'train_ix': train_ix,
    'test_ix': test_ix,
    'model_id': model_id,
}

with open(f'saved_models/gpt_{model_id}.cpkl', 'wb') as f:
    cloudpickle.dump(saving, f)

### Test

with open('saved_models/test_batch.cpkl', 'rb') as f:
    input_test, target_test, padding_test = cloudpickle.load(f)

with torch.no_grad():
    out = model(input_test.to(device), padding_test.to(device))

c_mean, b, o = out[0].cpu(), out[1].cpu(), out[2].cpu()

plotsize(4,4)
ix = 0
x_i = 0
y_i = 1
sns.scatterplot(
    x = input_test[ix, padding_test[0] == 0, :].numpy()[:, x_i],
    y = input_test[ix, padding_test[0] == 0, :].numpy()[:, y_i],
)
sns.scatterplot(
    x = c_mean[ix, padding_test[0] == 0, :].numpy()[:, x_i],
    y = c_mean[ix, padding_test[0] == 0, :].numpy()[:, y_i],
)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.savefig(f'saved_plots/{model_id}.png')


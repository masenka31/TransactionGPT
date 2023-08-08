import os
import sys
os.chdir("/Users/misa/Documents/TransactionGPT/")
sys.path.append('/Users/misa/Documents/TransactionGPT/python')

from src.process_data import IBMDataset
from src.process_data import train_val_test_indexes
from src.process_data import train_val_test_datasets
from src.process_data import fit_onehot_encoders
from src.process_data import positive_negative_indexes
from src.process_data import CustomDataset
from src.process_data import CustomUpsampleDataset
from src.models import FraudDetector
from src.utils import get_probabilities
from src.utils import enhance_minibatch
from src.results import Results
from src.utils import Hyperparameters

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import uuid
import argparse

## External arguments

parser = argparse.ArgumentParser(description="Input parameters.")
parser.add_argument('data_seed', type=int, help='seed for data split', default=1)
parser.add_argument('hyperparameter_seed', type=int, help='seed for hyperparameter sampling', default=1)
args = parser.parse_args()

### Data

ibm = IBMDataset()
data = ibm.process_dataset()

data_seed = args.data_seed
hyperparameter_seed = args.hyperparameter_seed
train_ix, val_ix, test_ix = train_val_test_indexes(data, random_state=data_seed)

train_data = data.loc[train_ix]
val_data = data.loc[val_ix]
test_data = data.loc[test_ix]

oh_customer_state, oh_merchant_state, oh_mcc, oh_chip = fit_onehot_encoders(train_data)
train_dataset, val_dataset, test_dataset = train_val_test_datasets(data, train_ix, val_ix, test_ix, oh_customer_state, oh_merchant_state, oh_mcc, oh_chip)

pos_idx, neg_idx = positive_negative_indexes(train_data)

### Hyperparameters

hyperparameters = Hyperparameters(hyperparameter_seed)
parameters = hyperparameters.sample_hyperparameters()
print(parameters)

input_size = train_dataset[0][0].shape[-1]
hidden_size = parameters.hidden_size
num_layers = parameters.num_layers
bidirectional = parameters.bidirectional
output_size = 2
learning_rate = parameters.learning_rate
ratio = parameters.ratio
upsampling_threshold = parameters.upsampling_threshold
batch_size = parameters.batch_size

### Model, loss, optimizer

# model
model = FraudDetector(input_size, hidden_size, output_size, num_layers, bidirectional)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# loss
class_weights = torch.tensor([1.0, ratio], dtype=torch.float32).to(device) # device can be 'cuda' or 'cpu'
loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

### DataLoaders

tmp1 = CustomUpsampleDataset(train_dataset, pos_idx, neg_idx, length=10_000, threshold=parameters.upsampling_threshold)
train_upsample_loader = DataLoader(tmp1, batch_size=parameters.batch_size, shuffle=True)

tmp11 = CustomDataset(train_dataset)
train_loader = DataLoader(tmp11, batch_size=parameters.batch_size, shuffle=False)

tmp2 = CustomDataset(val_dataset)
val_loader = DataLoader(tmp2, batch_size=parameters.batch_size, shuffle=False)

tmp3 = CustomDataset(test_dataset)
test_loader = DataLoader(tmp3, batch_size=parameters.batch_size, shuffle=False)

# Test that it works

for inputs, labels in train_loader:
    break

out = model(inputs.to(device))
loss_fn(out, labels.long().to(device))

# Initialization of results calculations

train_losses = []
train_accuracies = []

# Define the training loop

def train(model, optimizer, train_dataset, parameters, num_epochs, train_losses, train_accuracies, *, device):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        last_inputs, last_labels = 1, 1
        
        tmp = CustomUpsampleDataset(train_dataset, pos_idx, neg_idx, length=10_000, threshold=parameters.upsampling_threshold)
        train_loader = DataLoader(tmp, batch_size=parameters.batch_size, shuffle=True)

        for inputs, labels in train_loader:
            if parameters.enhance:
                if type(last_inputs) != int:
                    inputs, labels = enhance_minibatch(model, last_inputs, last_labels, inputs, labels)

                last_inputs = inputs.to(torch.device("cpu")).clone()
                last_labels = labels.to(torch.device("cpu")).clone()

            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum()
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Print training and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")


#  TODO: validation early stopping?
num_epochs = 10
train(model, optimizer, train_dataset, parameters, num_epochs, train_losses, train_accuracies, device = device)

# Calculate the predictions
break_at = float("inf")
# break_at = 200
ytrue_tr, yhat_tr = get_probabilities(model.to(torch.device("cpu")), train_dataset, break_at=break_at)
ytrue_val, yhat_val = get_probabilities(model.to(torch.device("cpu")), val_dataset, break_at=break_at)
ytrue_ts, yhat_ts = get_probabilities(model.to(torch.device("cpu")), test_dataset, break_at=break_at)

### Results
id = uuid.uuid1()

results = Results(ytrue_tr, yhat_tr, ytrue_val, yhat_val, ytrue_ts, yhat_ts)
results.print_confusion_matrix("val")
# results.print_confusion_matrices()
# results.plot_probabilities("train")
metrics_train = results.calculate_metrics("train")
metrics_val = results.calculate_metrics("val")
metrics_test = results.calculate_metrics("test")
results_dictionary = {
    # "results": results,
    "id": id,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "bidirectional": bidirectional,
    "learning_rate": learning_rate,
    "ratio": ratio,
    "upsampling_threshold": upsampling_threshold,
    "batch_size": batch_size
} | metrics_train | metrics_val | metrics_test

with open(f"data/results/IBM/fraud/rnn_baseline/{id}.pickle", "wb") as f:
    pickle.dump(results_dictionary, f)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f'data/results/IBM/fraud/rnn_baseline/{id}.pth')

# This is how to load it all back in
# checkpoint = torch.load('model_checkpoint.pth')

# from copy import copy, deepcopy
# modelx = FraudDetector(input_size, hidden_size, output_size, num_layers, bidirectional)
# modelx.load_state_dict(checkpoint['model_state_dict'])

# optimizerx = optim.Adam(modelx.parameters(), lr=learning_rate)
# optimizerx.load_state_dict(checkpoint['optimizer_state_dict'])

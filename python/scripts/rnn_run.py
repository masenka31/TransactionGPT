from python.src.process_data import IBMDataset
from python.src.process_data import train_val_test_indexes
from python.src.process_data import train_val_test_datasets
from python.src.process_data import fit_onehot_encoders
from python.src.process_data import positive_negative_indexes
from python.src.process_data import CustomDataset
from python.src.process_data import CustomUpsampleDataset
from python.src.models import FraudDetector
from python.src.utils import calculate_accuracy_and_f1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cloudpickle

ibm = IBMDataset()
data = ibm.process_dataset()

train_ix, val_ix, test_ix = train_val_test_indexes(data)

train_data = data.loc[train_ix]
val_data = data.loc[val_ix]
test_data = data.loc[test_ix]

oh_customer_state, oh_merchant_state = fit_onehot_encoders(train_data)
train_dataset, val_dataset, test_dataset = train_val_test_datasets(data, train_ix, val_ix, test_ix, oh_customer_state, oh_merchant_state)

pos_idx, neg_idx = positive_negative_indexes(train_data)

# Hyperparameters

input_size = train_dataset[0][0].shape[-1]
hidden_size = 64
num_layers = 3
bidirectional = False
output_size = 2
learning_rate = 0.0005
ratio = 0.1

batch_size = 32

# Model, loss, optimizer

model = FraudDetector(input_size, hidden_size, output_size, num_layers, bidirectional)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# bce = nn.BCELoss(reduction='none')

# def loss_fn(x, y, ratio=100):
#     ind_losses = bce(x, y)
#     scaled_losses = ind_losses * (y * (ratio - 1) + 1) / ratio
#     return torch.mean(scaled_losses)

# def get_loss_fn(class_weights, reduction='mean'):
#     return nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

# Usage:
class_weights = torch.tensor([1.0, ratio], dtype=torch.float32).to(device) # device can be 'cuda' or 'cpu'
loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

# DataLoaders

tmp1 = CustomUpsampleDataset(train_dataset, pos_idx, neg_idx, length=10_000, threshold=0.9)
train_upsample_loader = DataLoader(tmp1, batch_size=batch_size, shuffle=True)

tmp11 = CustomDataset(train_dataset)
train_loader = DataLoader(tmp11, batch_size=batch_size, shuffle=False)

tmp2 = CustomDataset(val_dataset)
val_loader = DataLoader(tmp2, batch_size=batch_size, shuffle=False)

tmp3 = CustomDataset(test_dataset)
test_loader = DataLoader(tmp3, batch_size=batch_size, shuffle=False)

# Test that it works

for inputs, labels in train_upsample_loader:
    break

out = model(inputs.to(device))
loss_fn(out, labels.long().to(device))

# Initialization of results calculations

max_validation_f1 = [0]
train_losses = []
train_accuracies = []

# Define the training loop

def train(model, optimizer, train_dataset, val_dataset, num_epochs, max_validation_f1, train_losses, train_accuracies, *, ratio=100, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        last_inputs, last_labels = 1, 1
        
        tmp = CustomUpsampleDataset(train_dataset, pos_idx, neg_idx, length=10_000, threshold=0.5)
        train_loader = DataLoader(tmp, batch_size=batch_size, shuffle=True)

        for inputs, labels in train_loader:
            if type(last_inputs) != int:
                inputs, labels = enhance_minibatch(model, last_inputs, last_labels, inputs, labels)

            last_inputs = inputs.to(torch.device("cpu")).clone()
            last_labels = labels.to(torch.device("cpu")).clone()

            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            # loss = loss_fn(outputs, labels, ratio=ratio)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            # train_correct += ((outputs >= 0.5).squeeze().long() == labels).sum().item()
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum()
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # # Print training and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        

#  TODO: validation early stopping?
num_epochs = 100
train(model, optimizer, train_dataset, val_dataset,
      num_epochs, max_validation_f1, train_losses, train_accuracies,
      ratio = ratio, device = device
)

from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from python.src.utils import get_predictions, get_probabilities, color_confusion_matrix
from python.src.utils import plot_roc_pr, calculate_optimal_threshold
from python.src.utils import print_confusion_matrix
from python.src.utils import plot_probabilities

# Calculate the predictions
break_at = float("inf")
break_at = 100
ytrue_tr, yhat_tr = get_probabilities(model.to(torch.device("cpu")), train_dataset, break_at=break_at)
ytrue_val, yhat_val = get_probabilities(model.to(torch.device("cpu")), val_dataset, break_at=break_at)
ytrue_ts, yhat_ts = get_probabilities(model.to(torch.device("cpu")), test_dataset, break_at=break_at)

from python.src.results import Results
results = Results(ytrue_tr, yhat_tr, ytrue_val, yhat_val, ytrue_ts, yhat_ts)
results.plot_roc_pr("train")

results.print_confusion_matrices()

# Look at the ROC curve and PR curve
plot_roc_pr(ytrue_tr, yhat_tr)

# Find the optimal threshold for F1 score
optimal_threshold = calculate_optimal_threshold(ytrue_val, yhat_val)

# Print validation and test confusion matrices
cm_val = confusion_matrix(ytrue_val, np.array(yhat_val) > optimal_threshold)
print_confusion_matrix(cm_val, 'Validation results')
cm_test = confusion_matrix(ytrue_ts, np.array(yhat_ts) > optimal_threshold)
print_confusion_matrix(cm_test, 'Test results')

# Plotting of probability distributions for legitimate and fraud sequences
plot_probabilities(yhat_tr, ytrue_tr)
plot_probabilities(yhat_val, ytrue_val)
plot_probabilities(yhat_ts, ytrue_ts)

# Create a combined plot for train, validation and test samples
plt.subplot(3,1,1)
plot_probabilities(yhat_tr, ytrue_tr, show=False)
plt.subplot(3,1,2)
plot_probabilities(yhat_val, ytrue_val, show=False)
plt.subplot(3,1,3)
plot_probabilities(yhat_ts, ytrue_ts, show=False)
# plt.show()
plt.savefig("plot2.pdf")

### Testing

last_inputs = inputs
last_labels = labels
next_inputs = inputs
next_labels = labels

def enhance_minibatch(model, last_inputs, last_labels, next_inputs, next_labels):
    """
    Enhance the next minibatch with
    - negative sample with the highest score
    - positive sample with the lowest score

    This should help the model focus more on the edge cases and try to separate
    the two categories better.
    """
    last_inputs = last_inputs.to(torch.device("cpu"))
    last_labels = last_labels.to(torch.device("cpu"))
    next_inputs = next_inputs.to(torch.device("cpu"))
    next_labels = next_labels.to(torch.device("cpu"))

    with torch.no_grad():
        outputs = model(last_inputs.to(device))
        outputs = outputs.to(torch.device("cpu"))

    if len(last_labels == 0) == 0 or len(last_labels == 1) == 0:
        return next_inputs, next_labels

    # negative_outputs = outputs[last_labels == 0]
    negative_outputs = outputs[last_labels == 0][:, 1]
    negative_ix = torch.argmax(negative_outputs)
    negative = last_inputs[last_labels == 0, :, :][negative_ix, :, :].unsqueeze(0)

    # positive_outputs = outputs[last_labels == 1]
    positive_outputs = outputs[last_labels == 1][:, 1]
    positive_ix = torch.argmin(positive_outputs)
    positive = last_inputs[last_labels == 1, :, :][positive_ix, :, :].unsqueeze(0)

    inputs = torch.cat((next_inputs, negative, positive))
    labels = torch.cat((next_labels, torch.tensor([0]), torch.tensor([1])))

    return inputs, labels
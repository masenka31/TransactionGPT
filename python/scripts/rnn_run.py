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
hidden_size = 256
num_layers = 3
bidirectional = False
output_size = 1
learning_rate = 0.001

batch_size = 64

# Model, loss, optimizer

model = FraudDetector(input_size, hidden_size, output_size, num_layers, bidirectional)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

bce = nn.BCELoss(reduction='none')

def loss_fn(x, y, ratio=100):
    ind_losses = bce(x, y)
    scaled_losses = ind_losses * (y * (ratio - 1) + 1) / ratio
    return torch.mean(scaled_losses)

# DataLoaders

tmp1 = CustomUpsampleDataset(train_dataset, pos_idx, neg_idx, length=10_000, threshold=0.1)
train_upsample_loader = DataLoader(tmp1, batch_size=batch_size, shuffle=True)

tmp11 = CustomDataset(train_dataset)
train_loader = DataLoader(tmp11, batch_size=batch_size, shuffle=False)

tmp2 = CustomDataset(val_dataset)
val_loader = DataLoader(tmp2, batch_size=batch_size, shuffle=False)

tmp3 = CustomDataset(test_dataset)
test_loader = DataLoader(tmp3, batch_size=batch_size, shuffle=False)

# Initialization of results calculations

max_validation_f1 = [0]
train_losses = []
train_accuracies = []

# Define the training loop

def train(model, optimizer, train_loader, val_dataset, num_epochs, max_validation_f1, train_losses, train_accuracies, ratio=100):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels, ratio=ratio)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            train_correct += ((outputs >= 0.5).squeeze().long() == labels).sum().item()
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # # Print training and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        
        # Get validation scores
        if (epoch + 1) % 10 == 1 and epoch > 10:
            f1, acc = calculate_accuracy_and_f1(model, val_dataset, break_at=20, return_values=True)
            if f1 > max(max_validation_f1):
                max_validation_f1.append(f1)

#  TODO: validation early stopping?
num_epochs = 100
train(model, optimizer, train_upsample_loader, val_dataset,
      num_epochs, max_validation_f1, train_losses, train_accuracies,
      ratio=5
)

from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt

from python.src.utils import get_predictions, color_confusion_matrix

# Calculate the predictions
break_at = float("inf")
ytrue_tr, yhat_tr = get_predictions(model, train_dataset, break_at=break_at)
ytrue_val, yhat_val = get_predictions(model, val_dataset, break_at=break_at)
ytrue_ts, yhat_ts = get_predictions(model, test_dataset, break_at=break_at)

# ROC curve
fpr, tpr, thresholds = roc_curve(ytrue_tr, yhat_tr)
roc_auc = auc(fpr, tpr) # compute Area Under the Curve

plt.subplot(1,2,1)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# PR curve
precision, recall, thresholds = precision_recall_curve(ytrue_tr, yhat_tr)
pr_auc = auc(recall, precision) # compute Area Under the Curve

# plt.figure()
plt.subplot(1,2,2)
plt.plot(recall, precision, color='b', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()

# find the optimal threshold for F1 score
precision, recall, thresholds = precision_recall_curve(ytrue_val, yhat_val)

b = precision + recall
precision = precision[b != 0]
recall = recall[b != 0]
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print("Threshold value is:", optimal_threshold)
print('Maximum F1 Score is', round(max(f1_scores) * 100, 2))

from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(ytrue, np.array(yhat) > 0.999)

cm = confusion_matrix(ytrue_val, np.array(yhat_val) > optimal_threshold)

cm_labeled = pd.DataFrame({
    'Predicted Negative': {'Actual Negative': f'TN: {cm[0,0]}', 'Actual Positive': f'FN: {cm[1,0]}'},
    'Predicted Positive': {'Actual Negative': f'FP: {cm[0,1]}', 'Actual Positive': f'TP: {cm[1,1]}'},
})
print('Validation results')
print(cm_labeled)

cm = confusion_matrix(ytrue_ts, np.array(yhat_ts) > optimal_threshold)

cm_labeled = pd.DataFrame({
    'Predicted Negative': {'Actual Negative': f'TN: {cm[0,0]}', 'Actual Positive': f'FN: {cm[1,0]}'},
    'Predicted Positive': {'Actual Negative': f'FP: {cm[0,1]}', 'Actual Positive': f'TP: {cm[1,1]}'},
})
print('Test results')
print(cm_labeled)


pr = cm[1,1] / (cm[1,1] + cm[0,1])
rc = cm[1,1] / (cm[1,0] + cm[1,1])
f1 = 2 * pr * rc / (pr + rc)
acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
print(f"F1 score: {round(f1, 4)}")
print(f"Accuracy: {round(acc, 4)}")

cm_labeled = cm_labeled.style.applymap(color_confusion_matrix)

from IPython.display import display
display(cm_labeled)

plt.scatter(thresholds[:-1], f1_scores, s=1)
plt.xlim(0.9,1)
plt.show()

import seaborn as sns
# sns.histplot(yhat, bins=100, stat='density')
# plt.subplot(1,2,1)
def plot_probabilities(yhat, ytrue):
    sns.histplot(np.array(yhat)[~np.array(ytrue, dtype=bool)], stat='density', color='green', label='Legit', binwidth=0.01, alpha=0.5)
    # plt.legend()
    # plt.subplot(1,2,2)
    sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], stat='density', color='red', label='Fraud', binwidth=0.01, alpha=0.5)
    plt.legend()
    plt.ylim(0,10)
    plt.show()

def plot_fraud_probabilities(yhat, ytrue):
    sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], color='red', label='Fraud', binwidth=0.01)
    plt.legend()
    plt.ylim(0,30)
    plt.show()

plot_probabilities(yhat_val, ytrue_val)
plot_probabilities(yhat_ts, ytrue_ts)
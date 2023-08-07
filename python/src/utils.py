from collections import namedtuple
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Hyperparameters():
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.hidden_size = [16, 32, 64, 128, 256]
        self.num_layers = [1, 2, 3]
        self.bidirectional = [True, False]
        self.learning_rate = [0.005, 0.001, 0.0005, 0.0001]
        self.ratio = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]
        self.upsampling_threshold = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.batch_size = [16, 32, 64, 128]

    def sample_hyperparameters(self):
        if self.seed is not None:
            random.seed(self.seed)

        Sample = namedtuple('Sample', ['hidden_size', 'num_layers', 'bidirectional', 'learning_rate', 'ratio', 'upsampling_threshold', 'batch_size'])

        hidden_size = random.choice(self.hidden_size)
        num_layers = random.choice(self.num_layers)
        bidirectional = random.choice(self.bidirectional)
        learning_rate = random.choice(self.learning_rate)
        ratio = random.choice(self.ratio)
        upsampling_threshold = random.choice(self.upsampling_threshold)
        batch_size = random.choice(self.batch_size)

        return Sample(hidden_size, num_layers, bidirectional, learning_rate, ratio, upsampling_threshold, batch_size)


def get_binary_predictions(model, dataset, break_at=100):
    model.eval()
    ytrue = []
    yhat = []
    tmp = 0
    for i in tqdm(range(len(dataset))):
        x, y = dataset.get_group_all_sequences(i)
        with torch.no_grad():
            yh = model(torch.tensor(x, dtype=torch.float32))
            ytrue.extend(y)
            yhat.extend(yh.detach().numpy())
            tmp += 1
            if tmp > break_at:
                break

    return ytrue, yhat

def get_probabilities(model, dataset, break_at=100):
    model.eval()
    ytrue = []
    yhat = []
    tmp = 0
    for i in tqdm(range(len(dataset))):
        x, y = dataset.get_group_all_sequences(i)
        with torch.no_grad():
            yh = nn.Softmax(dim=1)(model(torch.tensor(x, dtype=torch.float32)))[:, 1]
            ytrue.extend(y)
            yhat.extend(yh.detach().numpy())
            tmp += 1
            if tmp > break_at:
                break

    return ytrue, yhat

# def get_predictions(model, dataloader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             predictions = model(inputs).squeeze()
#             preds = torch.round(torch.sigmoid(predictions))
#             all_preds.extend(preds.detach().numpy())
#             all_labels.extend(labels.detach().numpy())
#     return all_labels, all_preds

def color_confusion_matrix(val):
    if 'TN' in val or 'TP' in val:
        color = 'green'
    else:
        color = 'red'
    return 'color: %s' % color

def calculate_accuracy_and_f1(model, dataset, break_at=100, return_values=False) -> tuple:
    """
    Calculates accuracy and F1 score for specified number of samples in `dataset`.

    returns: f1, acc
    """
    model.eval()
    ytrue = []
    yhat = []
    tmp = 0
    for i in tqdm(range(len(dataset))):
        x, y = dataset.get_group_all_sequences(i)
        with torch.no_grad():
            yh = model(torch.tensor(x, dtype=torch.float32))
            ytrue.extend(y)
            yhat.extend(yh.detach().numpy())
            tmp += 1
            if tmp > break_at:
                break

    f1 = f1_score(ytrue, np.round(yhat))
    acc = accuracy_score(ytrue, np.round(yhat))
    print('-'* 40)
    print('Scores')
    print('   - F1 score:', round(f1, 4))
    print('   - Accuracy:', round(acc, 4))
    print('-'* 40)
    if return_values:
        return f1, acc
    

def plot_roc_pr(ytrue, yhat, *, show=True):
    # ROC curve
    fpr, tpr, thresholds = roc_curve(ytrue, yhat)
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
    precision, recall, thresholds = precision_recall_curve(ytrue, yhat)
    pr_auc = auc(recall, precision) # compute Area Under the Curve

    # plt.figure()
    plt.subplot(1,2,2)
    plt.plot(recall, precision, color='b', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    
    if show:
        plt.show()


def calculate_optimal_threshold(ytrue, yhat):
    precision, recall, thresholds = precision_recall_curve(ytrue, yhat)

    b = precision + recall
    precision = precision[b != 0]
    recall = recall[b != 0]
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    print('Maximum F1 Score is', round(max(f1_scores) * 100, 2))

    return optimal_threshold


def print_confusion_matrix(cm, title='Validation results'):

    cm_labeled = pd.DataFrame({
        'Predicted Negative': {'Actual Negative': f'TN: {cm[0,0]}', 'Actual Positive': f'FN: {cm[1,0]}'},
        'Predicted Positive': {'Actual Negative': f'FP: {cm[0,1]}', 'Actual Positive': f'TP: {cm[1,1]}'},
    })
    print('-'*60)
    print(title)
    print('-'*60)
    print(cm_labeled)
    print('-'*60)

    pr = cm[1,1] / (cm[1,1] + cm[0,1])
    rc = cm[1,1] / (cm[1,0] + cm[1,1])
    f1 = 2 * pr * rc / (pr + rc)
    acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
    print(f"F1 score: {round(f1, 4)}")
    print(f"Accuracy: {round(acc, 4)}")
    print('-'*60)


def plot_probabilities(yhat, ytrue, *, show=True):
    sns.histplot(np.array(yhat)[~np.array(ytrue, dtype=bool)], stat='density', color='green', label='Legit', binwidth=0.01, alpha=0.5)
    # plt.legend()
    # plt.subplot(1,2,2)
    sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], stat='density', color='red', label='Fraud', binwidth=0.01, alpha=0.5)
    plt.legend()
    plt.ylim(0,10)
    if show:
        plt.show()

def plot_fraud_probabilities(yhat, ytrue, *, show=True):
    sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], color='red', label='Fraud', binwidth=0.01)
    plt.legend()
    plt.ylim(0,30)
    if show:
        plt.show()


def enhance_minibatch(model, last_inputs, last_labels, next_inputs, next_labels, *, device=torch.device("cpu")):
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
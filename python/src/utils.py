from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import torch
import numpy as np

def get_predictions(model, dataset, break_at=100):
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import pandas as pd


class Results:
    """
    Class to store and process true and predicted values for training, validation, and test data sets.
    Provides methods for plotting ROC and PR curves, calculating confusion matrices, and plotting histograms for predicted probabilities.
    """
    def __init__(self, ytrue_train, yhat_train, ytrue_val, yhat_val, ytrue_test, yhat_test):
        """
        Initialize the Results object with true and predicted values for training, validation, and test data.

        :param ytrue_train: Actual values for training data
        :param yhat_train: Predicted values for training data
        :param ytrue_val: Actual values for validation data
        :param yhat_val: Predicted values for validation data
        :param ytrue_test: Actual values for test data
        :param yhat_test: Predicted values for test data
        """
        self.ytrue_train = ytrue_train
        self.yhat_train = yhat_train
        self.ytrue_val = ytrue_val
        self.yhat_val = yhat_val
        self.ytrue_test = ytrue_test
        self.yhat_test = yhat_test

    def get_type(self, type):
        """
        Return the true and predicted values based on the specified type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :return: Tuple containing the true and predicted values
        """
        if type == "test":
            return self.ytrue_test, self.yhat_test
        elif type == "val" or type == "validation":
            return self.ytrue_val, self.yhat_val
        else:
            return self.ytrue_train, self.yhat_train

    def plot_roc_pr(self, type: str, *, show=True):
        """
        Plot ROC and Precision-Recall curves for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :param show: Whether to display the plot (default=True)
        """
        ytrue, yhat = self.get_type(type)

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

    def calculate_optimal_threshold(self, type: str, *, show=False):
        """
        Calculate the optimal threshold that maximizes the F1 score for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :return: Optimal threshold value
        """
        ytrue, yhat = self.get_type(type)
        precision, recall, thresholds = precision_recall_curve(ytrue, yhat)
        b = precision + recall
        precision = precision[b != 0]
        recall = recall[b != 0]
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        if show:
            print("Threshold value is:", optimal_threshold)
            print('Maximum F1 Score is', round(max(f1_scores) * 100, 2))

        return optimal_threshold

    def calculate_confusion_matrix(self, type: str, *, threshold_type="val"):
        """
        Calculate the confusion matrix using the optimal threshold for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test") for which to calculate the confusion matrix
        :param threshold_type: Type of data ("train", "val", "validation", or "test") used to calculate the optimal threshold (default="val")
        :return: Confusion matrix
        """
        optimal_threshold = self.calculate_optimal_threshold(threshold_type)
        ytrue, yhat = self.get_type(type)
        cm = confusion_matrix(ytrue, np.array(yhat) > optimal_threshold)
        return cm
    
    def calculate_metrics(self, type: str):
        """
        Calculate the precision, recall, F1 score, and accuracy based on the confusion matrix for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :return: Dictionary containing precision, recall, F1 score, and accuracy, each prefixed with the type (e.g., "precision_train")
        """
        cm = self.calculate_confusion_matrix(type)
        pr = cm[1,1] / (cm[1,1] + cm[0,1])
        rc = cm[1,1] / (cm[1,0] + cm[1,1])
        f1 = 2 * pr * rc / (pr + rc)
        acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
        return {
            f"precision_{type}": pr,
            f"recall_{type}": rc,
            f"f1_{type}": f1,
            f"accuracy_{type}": acc 
        }

    def print_confusion_matrix(self, type: str, title: str | None = None):
        """
        Print the confusion matrix along with precision, recall, F1 score, and accuracy for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :param title: Title for the printout (default='Validation results')
        """
        cm = self.calculate_confusion_matrix(type)
        cm_labeled = pd.DataFrame({
            'Predicted Negative': {'Actual Negative': f'TN: {cm[0,0]}', 'Actual Positive': f'FN: {cm[1,0]}'},
            'Predicted Positive': {'Actual Negative': f'FP: {cm[0,1]}', 'Actual Positive': f'TP: {cm[1,1]}'},
        })
        if title is not None:
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

    def print_confusion_matrices(self):
        self.print_confusion_matrix("train", "Train results")
        self.print_confusion_matrix("val", "Validation results")
        self.print_confusion_matrix("test", "Test results")

    def plot_probabilities(self, type: str, *, show=True):
        """
        Plot histograms of predicted probabilities for legit and fraud cases for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :param show: Whether to display the plot (default=True)
        """
        ytrue, yhat = self.get_type(type)
        sns.histplot(np.array(yhat)[~np.array(ytrue, dtype=bool)], stat='density', color='green', label='Legit', binwidth=0.01, alpha=0.5)
        sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], stat='density', color='red', label='Fraud', binwidth=0.01, alpha=0.5)
        plt.legend()
        plt.ylim(0,10)
        plt.xlim(0,1)
        if show:
            plt.show()

    def plot_fraud_probabilities(self, type: str, *, show=True):
        """
        Plot a histogram of predicted probabilities for fraud cases for the specified data type.

        :param type: Type of data ("train", "val", "validation", or "test")
        :param show: Whether to display the plot (default=True)
        """
        ytrue, yhat = self.get_type(type)
        sns.histplot(np.array(yhat)[np.array(ytrue, dtype=bool)], color='red', label='Fraud', binwidth=0.01)
        plt.legend()
        plt.ylim(0,30)
        if show:
            plt.show()
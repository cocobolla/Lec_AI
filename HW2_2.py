import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    root_path = r'C:\Users\USER\workspace\Lec\Lec_AI\HW2_data'
    file_name = 'scoredEX.csv'
    file_path = os.path.join(root_path, file_name)

    df = pd.read_csv(file_path)
    threshold_set = set(df['p_pred'])
    threshold_list = sorted(list(threshold_set))
    precision_list = []
    recall_list = []
    y_test = df['y_test']
    for t in threshold_list:
        y_pred = df['p_pred'] >= t
        y_pred *= 1  # Convert bool to integer
        precision = get_precision(df['y_test'], y_pred)
        recall = get_recall(y_test, y_pred)
        precision_list.append(precision)
        recall_list.append(recall)

    # 2-(1)
    plt.title("Recall - Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall_list, precision_list)
    plt.show()

    # 2 - (2)
    precision_np = np.array(precision_list)
    recall_np = np.array(recall_list)
    f1_np = 2 * (precision_np * recall_np) / (precision_np + recall_np)
    max_index = f1_np.argmax()
    optimal_t = threshold_list[max_index]
    print("Optimal Threshold: {}".format(optimal_t))

    # 2 - (3)
    y_pred = df['p_pred'] >= optimal_t
    y_pred *= 1
    accuracy = get_accuracy(y_test, y_pred)
    print("Accuracy: {:.3f}".format(accuracy))


def get_precision(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    return precision


def get_recall(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return recall


def get_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    return accuracy


if __name__ == '__main__':
    main()


from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, hamming_loss

def average_bal_acc_score(y_true1,y_pred1):
    sum = 0
    columns = y_pred1.columns
    for col in columns:
        c = balanced_accuracy_score(y_true1[col], y_pred1[col])
        sum += c
    return sum/len(y_pred1.columns)


def average_f1_score(y_true1,y_pred1):
    sum = 0
    columns = y_pred1.columns
    for col in columns:
        c = f1_score(y_true1[col], y_pred1[col])
        sum += c
    return sum/len(y_pred1.columns)


def average_acc_score(y_true1,y_pred1):
    sum = 0
    columns = y_pred1.columns
    for col in columns:
        c = accuracy_score(y_true1[col], y_pred1[col])
        sum += c
    return sum/len(y_pred1.columns)


def average_hamming_loss(y_true1,y_pred1):
    sum = 0
    columns = y_pred1.columns
    for col in columns:
        c = hamming_loss(y_true1[col], y_pred1[col])
        sum += c
    return sum/len(y_pred1.columns)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# accuracy score
def acc_score(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred)


# ROC AUC-score
def rocauc_score(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)
    return roc_auc_score(y_test,y_pred)


# confusion matrix
def confusion_matrix(y_test, predictions, classifier):
    cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()
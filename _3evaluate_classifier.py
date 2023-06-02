import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from _1features import features2Dataframe
import pickle
import pandas as pd
import os
from tqdm import tqdm

# accuracy score
def acc_score(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred)


# ROC AUC-score
def rocauc_score(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)
    return roc_auc_score(y_test,y_pred)


# confusion matrix
def display_confusion_matrix(y_test, predictions, classifier, title = "Confusion Matrix"):
    cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    disp.ax_.set_title(title)
    plt.show()


# function for classifying new images
def classify(img, mask):
    features = features2Dataframe(img, mask)

    classifier = pickle.load(open('final_classifier.sav', 'rb'))
    scalar = pickle.load(open('final_scalar.sav', 'rb'))

    features = scalar.transform(features)

    pred_label = classifier.predict(features)
    prob = classifier.predict_proba(features)
    return pred_label, prob


def main():
    metadata = pd.read_csv("Data/metadata.csv")
    features = pd.read_csv("Data/features.csv")

    df = pd.merge(metadata, features, left_on= "img_id", right_on= "image_names")
    COI = ["diagnostic","asymmetry","img_id"]
    df = df.loc[:,COI]
    df = df.dropna()
    cancers = {"BCC":1, "MEL":1, "SCC": 1, "ACK": 0, "NEV":0, "SEK":0}
    df["diagnostic"] = df["diagnostic"].replace(cancers)

    y_pred = []
    y_gt = []

    n = 30
    for img, gt in tqdm(zip(df["img_id"][:n], df["diagnostic"][:n])):
        im = plt.imread(os.path.join("Data/images", img))
        mask = plt.imread(os.path.join("Data/masks", img))
        y_pred.append(classify(im, mask)[0][0])
        y_gt.append(gt)

    print(accuracy_score(y_gt, y_pred))
    


if __name__ == "__main__":
    main()
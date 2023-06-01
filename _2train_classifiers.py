import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
METADATA_PATH = "Data/metadata.csv"
FEATURES_PATH = "Data/features.csv"
MODE = "all_data"

# loading and preparing data
metadata = pd.read_csv(METADATA_PATH)
features = pd.read_csv(FEATURES_PATH)

def train_test_val_split(X, y, train_size = 0.7, test_size = 0.2, val_size = 0.1):
    x_train, x_val, y_train, y_val = train_test_split(X,y, train_size= 1-val_size, shuffle=True, random_state = SEED)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = train_size / (1-val_size))
    return x_train, x_test, x_val, y_train, y_test, y_val


df = pd.merge(metadata, features, left_on= "img_id", right_on= "image_names")
COI = ["diagnostic", "age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "compactness", "multicolor_rate", "asymmetry"]
df = df.loc[:,COI]
ddf = df.dropna()
COIbool = ["itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed"]
d = {"False": 0, "True": 1, False: 0, True:1, "UNK": 0.5}
for col in COIbool:
    df[col] = df[col].replace(d)
df = df.dropna()

modes = {
    "features" : ["compactness", "multicolor_rate", "asymmetry"],
    "all_data" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "compactness", "multicolor_rate", "asymmetry"],
    "metadata" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed",],
}
mode = modes[MODE]

y = df["diagnostic"]
X = df[mode]

cancers = {"BCC":1, "MEL":1, "SCC": 1, "ACK": 0, "NEV":0, "SEK":0}
y = y.replace(cancers)


scalar = StandardScaler()
X = scalar.fit_transform(X)

X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X,y)


# KNN-classifiers
knnc = KNeighborsClassifier(n_neighbors=20)

n_vals = [i for i in range(1,164)]
n_vals = [i for i in range(1,int(len(X_train) ** 0.5))]
n_vals = [i for i in range(1,500)]
knn_classifiers = []

for i in n_vals:
    knnc = KNeighborsClassifier(n_neighbors=i)
    knnc.fit(X_train, y_train)
    knn_classifiers.append(knnc)

predicted_ys = [model.predict(X_test) for model in knn_classifiers]
accuracy_scores = [accuracy_score(y_test, y_pred) for y_pred in predicted_ys]
roc_auc_scores = [roc_auc_score(y_test,y_pred) for y_pred in predicted_ys]


for i, val in enumerate(n_vals):
   print(f"N = {val} \t Accuracy scores = {round(accuracy_scores[i], 4)} \t ROC AUC Scores = {round(roc_auc_scores[i],4)}")

import matplotlib.pyplot as plt
plt.scatter(n_vals, roc_auc_scores)
plt.show()

# logistic regression
clf  = LogisticRegression().fit(X = X_train, y = y_train)

pred = clf.predict(X_test)
print(clf.score(X_test, y_test))
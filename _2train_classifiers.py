import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# loading and preparing data
raw_df = pd.read_csv("metadata.csv")
features = pd.read_csv("features.csv")
COI = ["diagnostic", "age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed"]
COI2 = ["diagnostic", "age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "fitspatrick"]
df = raw_df.loc[:,COI]
df2 = raw_df.loc[:,COI2]
df = df.dropna()
df2 = df2.dropna()
COIbool = ["itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed"]
d = {"False": 0, "True": 1, False: 0, True:1, "UNK": 0.5}
for col in COIbool:
    df2[col] = df2[col].replace(d)
    df[col] = df[col].replace(d)
df = df.dropna()
df2 = df2.dropna()
y = df["diagnostic"]
X = df.drop("diagnostic",axis=1)
cancers = {"BCC":1, "MEL":1, "SCC": 1, "ACK": 0, "NEV":0, "SEK":0}
y = y.replace(cancers)


# KNN-classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, train_size=0.75)
knnc = KNeighborsClassifier(n_neighbors=20)


# logistic regression
scalar = StandardScaler()
X_norm = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y,test_size=0.25, train_size=0.75)
clf  = LogisticRegression().fit(X = X_train, y = y_train)
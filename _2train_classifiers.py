import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from _3evaluate_classifier import display_confusion_matrix

SEED = 42
METADATA_PATH = "Data/metadata.csv"
FEATURES_PATH = "Data/features.csv"
X_MODE = "all_data"
Y_MODE = "cancers"


def train_test_val_split(X, y, train_size = 0.7, test_size = 0.2, val_size = 0.1, seed = 100):
    x_train, x_val, y_train, y_val = train_test_split(X,y, train_size= 1-val_size, shuffle=True, random_state = SEED)
    if seed:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = train_size / (1-val_size), shuffle = True, random_state=seed)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = train_size / (1-val_size), shuffle = True)
    return x_train, x_test, x_val, y_train, y_test, y_val



# loading and preparing data
metadata = pd.read_csv(METADATA_PATH)
features = pd.read_csv(FEATURES_PATH)
df = pd.merge(metadata, features, left_on= "img_id", right_on= "image_names")
COI = ["diagnostic", "age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed",
        "compactness", "multicolor_rate", "asymmetry",
          "avg_red_channel", "avg_green_channel", "avg_blue_channel",
          "average_hue", "average_saturation", "average_value"]
df = df.loc[:,COI]
df = df.dropna()
COIbool = ["itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed"]
d = {"False": 0, "True": 1, False: 0, True:1, "UNK": 0.5}
for col in COIbool:
    df[col] = df[col].replace(d)


df["random"] = [np.random.randint(0,100) for _ in range(len(df["age"]))]

x_modes = {
    "features" : ["compactness", "multicolor_rate", "asymmetry"],
    "features_cols" : ["compactness", "multicolor_rate", "asymmetry", "avg_red_channel", "avg_green_channel", "avg_blue_channel"],
    "features_hsv" : ["compactness", "multicolor_rate", "asymmetry", "average_hue", "average_saturation", "average_value"],
    "all_data" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "compactness", "multicolor_rate", "asymmetry", "avg_red_channel", "avg_green_channel", "avg_blue_channel","average_hue", "average_saturation", "average_value"],
    "all_data_b" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "compactness", "multicolor_rate", "asymmetry", "avg_red_channel", "avg_green_channel", "avg_blue_channel","average_hue", "average_saturation", "average_value"],
    "no_metadata" : ["compactness", "multicolor_rate", "asymmetry", "avg_red_channel", "avg_green_channel", "avg_blue_channel","average_hue", "average_saturation", "average_value"],
    "metadata" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed",],
    "metadata_features" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed","compactness", "multicolor_rate", "asymmetry"],
    "metadata_rgb" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "avg_red_channel", "avg_green_channel", "avg_blue_channel"],
    "metadata_hsv" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "average_hue", "average_saturation", "average_value"],
    "random" : ["age", "itch", "grew", "hurt", "changed", "bleed", "elevation", "compactness", "multicolor_rate", "asymmetry", "avg_red_channel", "avg_green_channel", "avg_blue_channel","average_hue", "average_saturation", "average_value", "random", "biopsed"],
}

y_modes = {
    "cancers" : {"BCC":1, "MEL":1, "SCC": 1, "ACK": 0, "NEV":0, "SEK":0},
    "nevus" : {"BCC":0, "MEL":0, "SCC": 0, "ACK": 0, "NEV":1, "SEK":0},
    "all" : {"BCC":1, "MEL":2, "SCC": 3, "ACK": 4, "NEV":5, "SEK":6},
}

x_mode = x_modes[X_MODE]
y_mode = y_modes[Y_MODE]

def dataset(df, x_mode, y_mode, Pca = False, StandardScalar = False, seed = 30):
    y = df["diagnostic"]
    X = df[x_mode]
    y = y.replace(y_mode)

    if StandardScalar:
        scalar = StandardScaler()
        X = scalar.fit_transform(X)

    if Pca:
        pca = PCA(4)
        pca.fit(X)
        X = pca.transform(X)


    return train_test_val_split(X,y, seed = seed)

def k_datasets(df, x_mode, y_mode, Pca = False, StandardScalar = False, k = 10):
    return [dataset(df, x_mode, y_mode, Pca = False, StandardScalar = False) for _ in range(k)]

def one_out_datasets(df, x_mode, y_mode, Pca = False, StandardScalar = False, k = 10):
    x_one_out = []
    for col in x_mode:
        x_one_out.append([x for x in x_mode if x != col])

    datasets = []
    for mode in x_one_out:
        datasets.append(k_datasets(df, mode, y_mode, Pca = Pca, StandardScalar = StandardScalar, k = k))

    return datasets

# KNN-classifiers

def generate_knn(X_train, X_test, y_train, y_test, k = 5, returnP = False):
    knnc = KNeighborsClassifier(n_neighbors = k)
    knnc.fit(X_train, y_train)

    pred = knnc.predict(X_test)
    # display_confusion_matrix(y_test, pred, knnc)
    accuracy =  f1_score(y_test, pred)

    if returnP:
        return pred
    
    return accuracy
    


def knn_curve(X_train, X_test, y_train, y_test, k_lim = 42):
    n_vals = [i for i in range(1,k_lim)]
    knn_classifiers = []

    for i in n_vals:
        knnc = KNeighborsClassifier(n_neighbors=i)
        knnc.fit(X_train, y_train)
        knn_classifiers.append(knnc)

    predicted_ys = [model.predict(X_test) for model in knn_classifiers]
    accuracy_scores = [accuracy_score(y_test, y_pred) for y_pred in predicted_ys]
    f1_scores = [f1_score(y_test, y_pred) for y_pred in predicted_ys]
    # roc_auc_scores = [roc_auc_score(y_test,y_pred) for y_pred in predicted_ys]


    # for i, val in enumerate(n_vals):
    #    print(f"N = {val} \t Accuracy scores = {round(accuracy_scores[i], 4)} \t ROC AUC Scores = {round(roc_auc_scores[i],4)}")
    # for i, val in enumerate(n_vals):
    #     print(f"N = {val} \t Accuracy scores = {round(accuracy_scores[i], 4)}")

    plt.scatter(n_vals, f1_scores, c = "tab:orange")
    plt.title("F1 score for each k-value using our features and average RGB")
    plt.xlabel("K-Value")
    plt.ylabel("F1-Score")
    plt.show()


# logistic regression

def generate_logistic_regression(X_train, X_test, y_train, y_test, returnP = False):
    clf  = LogisticRegression(multi_class= "multinomial").fit(X = X_train, y = y_train)

    pred = clf.predict(X_test)

    # display_confusion_matrix(y_test, pred, clf)

    accuracy = accuracy_score(y_test, pred)

    accuracy = f1_score(pred, y_test)
    return accuracy

def plot_features(x, y, z = None, classes = None):
    colors = {1:'tab:blue', 0:'tab:orange', 2:'tab:pink', 3:'tab:purple', 4:'tab:cyan', 5:'tab:green', 6: 'tab:red'}
    
    c = "tab:blue"
    if classes:
        c = [colors[i] for i in classes]

    if not z:
        plt.scatter(x,y, c = c)
        plt.show()
        return

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(x, y, z, c = c, )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig.show()
    plt.show()


def one_outs_main(generate_classifier, k=500):
    datasets = one_out_datasets(df, x_mode, y_mode, Pca = False, StandardScalar = True, k=k)

    accuracies = {y: [] for y in x_mode}

    for data,n in tqdm(zip(datasets, x_mode)) :
        x_v = []
        y_v = []

        # print(x_mode)
        for d in data:
            X_train, X_test, X_val, y_train, y_test, y_val = d
            acc = generate_classifier(X_train, X_test, y_train, y_test)
            # print(f"Without {n} accuracy = {acc}")

            accuracies[n].append(acc)
            # x_v.append(n)
            # y_v.append(acc)
        
    baselines = []
    for _ in range(k):
        X_train, X_test, X_val, y_train, y_test, y_val = dataset(df, x_mode, y_mode, Pca = False, StandardScalar = True)
        baseline = generate_classifier(X_train, X_test, y_train, y_test)
        baselines.append(baseline)
    

    accuracies = pd.DataFrame(accuracies)

    return accuracies, np.median(baselines)

def confusion_matrices():
    for x_type in ["features_cols", "all_data"]:
        for y_type in ["cancer", "all"]:
            X_train, X_test, X_val, y_train, y_test, y_val = dataset(df, x_mode, y_mode, Pca= False, StandardScalar= True)
            
            for classifier in [generate_logistic_regression, generate_knn]:
                prediction = classifier(X_train, X_test, y_train, y_test, returnP = True)
                display_confusion_matrix(y_test, prediction)

def plot_boxplot(k = 500):
    accuracies_knn, baseline_knn = one_outs_main(generate_knn, k = k)
    accuracies_logistic, baseline_logistic_regression = one_outs_main(generate_logistic_regression, k = k)
    

    accuracies_knn["model"] = ["KNN" for _ in range(len(accuracies_knn))]
    accuracies_logistic["model"] = ["Logistic" for _ in range(len(accuracies_knn))]

    accuracies = pd.concat([accuracies_logistic, accuracies_knn])
    
    data=accuracies.melt(id_vars=['model'], value_vars= x_mode,
                 var_name='feature', value_name='value')
    print(data)

    # Sort the boxplots by median
    # order = accuracies.median().sort_values().index.to_list()

    # print(accuracies)
    c= sns.boxplot(x = "feature" , y= "value" ,data=data , hue= "model")

    # Add a vertical line at the height of the baseline
    # c.axhline(y=np.median(baseline_knn), color='tab:orange', linestyle='--')
    # c.axhline(y=np.median(baseline_logistic_regression), color='tab:blue', linestyle='--')

    # Rotate the labels by 45 degrees
    plt.xticks(rotation=30)

    plt.title("F1 score for model using all features taking one feature out")
    # Get handles for the legend
    handles, _ = c.get_legend_handles_labels()

    # Add labels for axhlines in the legend
    handles.extend([plt.axhline(np.median(baseline_knn), color='tab:orange', linestyle='--'), plt.axhline(np.median(baseline_logistic_regression), color='tab:blue', linestyle='--')])
    labels = ["Logistic", "Knn", "Baseline Knn", "Baseline logistic"]

    plt.legend(handles=handles, labels=labels, loc = "upper right")
    # plt.legend(handles = [a,b,c ], labels = ["Baseline Knn","Baseline Logistic", "a","b"])
    plt.show()


    # ax = plt.axes()
    # ax.hlines(baseline, -2, 30, colors = "tab:orange")
    # plt.vlines(x_v, min(y_v) -0.02, y_v, linestyle="dashed")
    # ax.scatter(x_v, y_v)
    # plt.xticks(rotation=45, ha='right')
    # plt.show()


def main(): 
    plot_boxplot()

    # X_train, X_test, X_val, y_train, y_test, y_val = dataset(df, x_mode, y_mode, Pca= False, StandardScalar= True)
    # knn_curve(X_train, X_test, y_train, y_test, k_lim=42)

    # confusion_matrices()



if __name__ == "__main__":
    main()
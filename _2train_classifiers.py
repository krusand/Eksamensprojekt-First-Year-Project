import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

SEED = 30
METADATA_PATH = "Data/metadata.csv"
FEATURES_PATH = "Data/features.csv"

X_MODES = {
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
}

Y_MODES = {
    "cancers" : {"BCC":1, "MEL":1, "SCC": 1, "ACK": 0, "NEV":0, "SEK":0},
    "nevus" : {"BCC":0, "MEL":0, "SCC": 0, "ACK": 0, "NEV":1, "SEK":0},
    "all" : {"BCC":1, "MEL":2, "SCC": 3, "ACK": 4, "NEV":5, "SEK":6},
}



# Data Utils

def train_val_test_split(X, y, train_size = 0.7, test_size = 0.2, val_size = 0.1, seed = SEED):
    """Split the data into 3 parts. The first split has a fixed seed to preserve untouched test data"""
    x_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 1-val_size, shuffle=True, random_state = 42)

    if seed:
        x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, train_size = train_size / (1-val_size), shuffle = True, random_state=seed)
    else:
        x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, train_size = train_size / (1-val_size), shuffle = True)
    
    return x_train, X_val, X_test, y_train, y_val, y_test


def load_data(metadata, features):
    """Loads metadata and features into a pandas dataframe

    Args:
        metadata : Metadata path
        features : Features path

    Returns:
        Combined Dataframe
    """
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

    return df

def dataset(df, x_mode, y_mode, StandardScalar = False, seed = None):
    """Generates a dataset split into X,y and train,val,test

    Args:
        StandardScalar (bool): Scale the dataset using StandardScalar
    """
    y = df["diagnostic"]
    X = df[x_mode]
    y = y.replace(y_mode)

    if StandardScalar:
        scalar = StandardScaler()
        X = scalar.fit_transform(X)


    return train_val_test_split(X,y, seed = seed)


def one_out_datasets(df, x_mode, y_mode, StandardScalar = False, k = 10):
    """Generates a 2x2 list of datasets where the columns are random shuffles and the rows are without a specific feature"""

    x_one_out = []
    for col in x_mode:
        x_one_out.append([x for x in x_mode if x != col])

    datasets = []
    for mode in x_one_out:
        datasets.append([dataset(df, x_mode, y_mode, StandardScalar = StandardScalar, seed= None) for _ in range(k)])

    return datasets




# Generate Predictions

def knn_predict(X_train, X_val, y_train, y_val, k = 5):
    """Train a KNN classifier and predict values"""
    knnc = KNeighborsClassifier(n_neighbors = k)
    knnc.fit(X_train, y_train)

    return knnc.predict(X_val)

def logistic_predict(X_train, X_val, y_train, y_val, returnP = False):
    """Train a Logistic Regression classifier and predict values"""
    clf  = LogisticRegression(multi_class= "multinomial").fit(X = X_train, y = y_train)
    return clf.predict(X_val)


# Score Models

def baseline_f1(df, x_mode, y_mode, classifier_predict, k):
    """Finds the median F1 score for k random shuffles"""
    baselines = []
    for i in range(k):
        X_train, X_val, X_test, y_train, y_val, y_test = dataset(df, x_mode, y_mode, StandardScalar = True, seed=i)
        pred = classifier_predict(X_train, X_val, y_train, y_val)
        score = f1_score(y_val, pred)
        baselines.append(score)

    return np.median(baselines)

def search_k_knn(k_lim, df, x_mode, y_mode, title, file):
    """Create a scatterplot for all F1 scores using k from 1 to k_lim"""
    x_mode = X_MODES[x_mode]
    y_mode = Y_MODES[y_mode]

    X_train, X_val, _, y_train, y_val,  _ = dataset(df, x_mode, y_mode, seed= SEED)
    n_vals = [i for i in range(1,k_lim +1)]
    knn_classifiers = []

    for i in n_vals:
        knnc = KNeighborsClassifier(n_neighbors=i)
        knnc.fit(X_train, y_train)
        knn_classifiers.append(knnc)

    predicted_ys = [model.predict(X_val) for model in knn_classifiers]
    f1_scores = [f1_score(y_val, y_pred) for y_pred in predicted_ys]

    plt.figure(figsize= (13,9))
    plt.scatter(n_vals, f1_scores, c = "tab:orange")
    plt.ylim(0.38, 0.68)
    plt.title(title, fontsize = 14)
    plt.xlabel("K-Value", fontsize = 14)
    plt.ylabel("F1-Score", fontsize = 14)
    # plt.show()
    plt.savefig(os.path.join("plots", file))


def one_out_scores(df, x_mode, y_mode, classifier_predict, k=500):
    """generate a dict of F1 scores where the values are F1 scores for random shuffles and for each key one feature is taken out"""

    datasets = one_out_datasets(df, x_mode, y_mode, StandardScalar = True, k=k)
    scores = {y: [] for y in x_mode}

    for data,n in tqdm(zip(datasets, x_mode)) :
        for d in data:
            X_train, X_val, X_test, y_train, y_val, y_test = d
            pred = classifier_predict(X_train, X_val, y_train, y_val)
            score = f1_score(y_val, pred)

            scores[n].append(score)
        
    baseline = baseline_f1(df, x_mode, y_mode, classifier_predict, k)
    
    

    scores = pd.DataFrame(scores)

    return scores, baseline 

def one_out_boxplot(df, x_mode, y_mode, repetitions, title, y_limit):
    """Creates boxplots over the F1 scores taking a single feature out"""

    # Generate the Scores
    x_mode = X_MODES[x_mode]
    y_mode = Y_MODES[y_mode]

    scores_knn, baseline_knn = one_out_scores(df, x_mode, y_mode, knn_predict, k = repetitions)
    scores_logistic, baseline_logistic_regression = one_out_scores(df, x_mode, y_mode, logistic_predict, k = repetitions)
    

    scores_knn["model"] = ["KNN" for _ in range(len(scores_knn))]
    scores_logistic["model"] = ["Logistic" for _ in range(len(scores_knn))]

    scores = pd.concat([scores_logistic, scores_knn])
    
    data=scores.melt(id_vars=['model'], value_vars= x_mode,
                 var_name='feature', value_name='value')

   
   #Plotting
    c= sns.boxplot(x = "feature" , y= "value" ,data=data , hue= "model")

    plt.xticks(rotation=30)
    plt.title(title, fontsize = 14)
    handles, _ = c.get_legend_handles_labels()
    handles.extend([plt.axhline(baseline_knn, color='tab:orange', linestyle='--'), plt.axhline(baseline_logistic_regression, color='tab:blue', linestyle='--')])
    labels = ["Logistic", "Knn", "Baseline Knn", "Baseline logistic"]

    plt.legend(handles=handles, labels=labels, loc = "upper right")
    plt.ylabel("F1-Score", fontsize = 14)
    plt.xlabel("")
    
    plt.ylim(y_limit)
    plt.show()

def create_confusion_matrix(df, x_mode, y_mode,classifier, title, file):
    """Plots a confusion matrix for one file"""

    x_mode = X_MODES[x_mode]
    y_mode = Y_MODES[y_mode]

    X_train, X_val, X_test, y_train, y_val, y_test = dataset(df, x_mode, y_mode, StandardScalar= True, seed=SEED)

    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_val)

    cm = confusion_matrix(y_val, pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot(text_kw={"fontsize" : 25})
    disp.ax_.set_title(title)

    plt.savefig(os.path.join("plots", file)) 
    plt.show()
    
def main(): 
    df = load_data(METADATA_PATH, FEATURES_PATH)

    # Plots:

    # Knn parameter search
    # search_k_knn(k_lim = 42, df = df, x_mode ="features_cols", y_mode = "cancers",
    #             title = "F1 score for each k-value using all features",
    #             file = "knn_curve_all.png")
    # search_k_knn(k_lim = 42, df = df, x_mode ="all_data", y_mode = "cancers", 
    #             title = "F1 score for each k-value using our features",
    #             file = "knn_curve_ours.png")

    # Boxplot the different F1 - Score one feature out
    # one_out_boxplot(df = df, x_mode = "no_metadata", y_mode= "cancers", repetitions = 200,
    #             title = "F1 score for model using our features taking one feature out",
    #             y_limit = (0.55, 0.95))
    # one_out_boxplot(df = df, x_mode = "all_data", y_mode= "cancers", repetitions = 200,
    #             title = "F1 score for model using all features taking one feature out",
    #             y_limit = (0.55, 0.95))
    
    # one_out_boxplot(df = df, x_mode = "features_cols", y_mode= "cancers", repetitions = 200,
    #             title = "F1 score for model using some features taking one feature out",
    #             y_limit = (0.5, 0.85))
    # one_out_boxplot(df = df, x_mode = "features_hsv", y_mode= "cancers", repetitions = 200,
    #             title = "F1 score for model using some features taking one feature out",
    #             y_limit = (0.5, 0.85))

    # Confusion matrices
    # create_confusion_matrix(df = df, x_mode = "all_data", y_mode= "cancers", 
    #                       classifier = KNeighborsClassifier(n_neighbors=5),
    #                       title = "Confusion matrix KNN all features",
    #                       file = "ConfusionKNNAll.png")
    
    # create_confusion_matrix(df = df, x_mode = "all_data", y_mode= "all", 
    #                       classifier = KNeighborsClassifier(n_neighbors=5),
    #                       title = "Confusion matrix KNN all features all labels",
    #                       file = "ConfusionKNNAllFeatures.png")
    
    # create_confusion_matrix(df = df, x_mode = "features_cols", y_mode= "cancers", 
    #                       classifier = KNeighborsClassifier(n_neighbors=5),
    #                       title = "Confusion matrix KNN our features",
    #                       file = "ConfusionKNNOur.png")
    
    # create_confusion_matrix(df = df, x_mode = "all_data", y_mode= "cancers", 
    #                       classifier = LogisticRegression(),
    #                       title = "Confusion matrix Logistic regression all features",
    #                       file = "ConfusionLogiAll.png")
    
    # create_confusion_matrix(df = df, x_mode = "features_cols", y_mode= "cancers", 
    #                       classifier = LogisticRegression(),
    #                       title = "Confusion matrix Logistic regression our features",
    #                       file = "ConfusionLogiOur.png")
    

    # F1 - scores
    # knn_predict()
    # logistic_predict()

    # Generate classifier:


if __name__ == "__main__":
    main()
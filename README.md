# Medical Imaging 
Welcome to our Github repository for our exam project in the First Year Project course.

The project focuses on segmentation, feature extraction and classification of medical images. In particular images of skin lesions. The goal of the project is to train a classifier, which is able to predict the probability that an unknown skin lesions is cancerous.

All of the extracted features can be found in the csv file `data/features.csv`. These features were created by running the python file `_1features.csv`. The file contains the extracted features for each image. 

By using the metadata and the extracted features, it is possible to train a series of classifiers. We chose to use KNN and Logistic Regression as classifiers. Training of the classifier is done using the file `2_train_classifiers.py`. Furthermore, evaluation metrics are found in the same file. The evaluation of the classifiers is done by calculating F1-score, ROC AUC-score, Accuracy score, and plotting a confusion matrix. To simulate these metrics with most confidence, we simulate different train test splits. The simulation is run 200 times and chooses different sets of the train and validation data. The test data is the same subset of skin lesions every time. These test runs are plotted using the `one_out_boxplots` function. 

To evaluate our chosen classifier, use the `_3evaluate_classifier.py` file. The file contains a function `classify` which takes an unknown image and mask as input and outputs the probability of the masked image being cancerous. The `classify` function is provided with the correct classifier saved in the root directory as a *.sav* file. `classify` imports the function `features2Dataframe` which extracts features from the provided image. 


The image data along with the metadata used for the project is provided by Mendeley Data and can be found [here](https://data.mendeley.com/datasets/zr7vgbcyr2/)
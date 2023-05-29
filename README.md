# Medical Imaging 
Welcome to our Github repository for our exam project in the First Year Project course.

The project focuses on segmentation, feature extraction and classification of medical images. In particular images of skin lesions. The goal of the project is to train a classifier, which is able to predict whether unknown skin lesions are cancerous or not. 

All of the extracted features can be found in the python file `1_features.py`. By applying the feature extractions the file `1_features.csv` is generated. The file contains the extracted features for the image data.

By using the metadata and the extracted features, it is possible to train a series of classifiers. These can be found in `2_train_classifiers.py`.

Finally the classifiers are evaluated by computing the accuracy score, ROC AOC score and the confusion matrix. The evaluation measurements are described in the file `3_evaluate_classifier.py`. The file also contains a function which takes an unknown image and mask as input and outputs the probability of the masked image being cancerous or not. This is done by measuring the features and predicting a label by using one of the classifiers from `2_train_classifiers.py`. 

The image data along with some metadata is provided by Mendeley Data and can be found here: https://data.mendeley.com/datasets/zr7vgbcyr2/. 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from modulos.yamnet_importation import *
from modulos.cough_classification import *
from pyAudioAnalysis import audioTrainTest as aT
import pandas as pd
import numpy as np
import os

wav_path = 'C:/Users/Guillem/Desktop/Bot_Telegram/Cough Audios/'

prediction_yamnet_indian_cough = []
prediction_svm_indian_cough = []
for i in np.arange(1,345):

    wav_file = os.path.join(wav_path, 'Indian_Cough/All_coughs/indian_cough ({}).wav'.format(i))
    prediction_yamnet = yamnet_classifier(wav_file)
    prediction_svm = aT.file_classification(wav_file, "cough_classifier/svm_cough", "svm")[1][0]
    prediction_yamnet_indian_cough.append(prediction_yamnet)
    prediction_svm_indian_cough.append(prediction_svm)

prediction_yamnet_cough = []
prediction_svm_cough = []
for i in np.arange(1,171):

    wav_file = os.path.join(wav_path, 'Cough/cough ({}).wav'.format(i))
    prediction_yamnet = yamnet_classifier(wav_file)
    prediction_svm = aT.file_classification(wav_file, "cough_classifier/svm_cough", "svm")[1][0]
    prediction_yamnet_cough.append(prediction_yamnet)
    prediction_svm_cough.append(prediction_svm)

prediction_yamnet_nocough = []
prediction_svm_nocough = []
for j in np.arange(1,174):

    wav_file = os.path.join(wav_path, 'No_Cough/No_Cough ({}).wav'.format(j))
    prediction_yamnet = yamnet_classifier(wav_file)
    prediction_svm = aT.file_classification(wav_file, "cough_classifier/svm_cough", "svm")[1][0]
    prediction_yamnet_nocough.append(prediction_yamnet)
    prediction_svm_nocough.append(prediction_svm)


y_true = np.append(np.repeat(True,344+170),np.repeat(False, 173))
y_pred_yamnet = prediction_yamnet_indian_cough + prediction_yamnet_cough + prediction_yamnet_nocough
y_pred_svm = prediction_svm_indian_cough + prediction_svm_cough + prediction_svm_nocough

len(y_true)
len(y_pred_yamnet)
len(y_pred_svm)

def Confusion_Matrix(y_true, y_predicted, binarized_true = False, binarized_pred = False, pred_prob = False):

    '''
    Generate a confusion matrix for binary classification. Additionally plots the ROC and AUC if the
    output of the model predicts probabilities
    @params:
        y_true          - A list of integers or strings for known classes
        y_predicted     - A list of integers, strings or probabilities for predicted classes
        binaried_true   - If the y_true are strings (FALSE,TRUE) converts to numerical [0,1]
        binaried_pred   - If the y_predicted are strings (FALSE,TRUE) converts to numerical [0,1]
        pred_prob       - If the predictions of the model are probabilities, then an optimal threshold can be compute
                          as well as plotting the ROC and AUC

    @return:
        confusion matrix
        classification Report
        optimal threshold
        AUC
        Plot of ROC
        y_true, y_predicted   (treated)

    @ Precision: What proportion of positive identifications was actually correct?
                 Our model has a precision of 0.5—in other words, when it predicts a recording is a cough, it is correct 50% of the time.

    @ Recall: What proportion of actual positives was identified correctly?
              Our model has a recall of 0.11—in other words, it correctly identifies 11% of all cough recordings.
    '''

    y_true = pd.Series(y_true, name = 'Actual')
    y_predicted = pd.Series(y_predicted, name='Predicted')

    if (binarized_true):
        y_true = y_true.map({True:1, False:0})

    if (binarized_pred):
        y_predicted = y_predicted.map({True:1, False:0})

    if (pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_predicted, pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print("\n __________ \n")
        print("Optimal Threshold: {} \n __________ \n".format(round(optimal_threshold,4)))

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (area = {})'.format(round(roc_auc,3)))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        y_predicted = pd.Series(np.where(y_predicted>=optimal_threshold, 1, 0), name='Predicted')


    df_confusion = pd.crosstab(y_true, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print("\n __________ \n")
    print("Confusion Matrix: \n __________ \n")
    print(df_confusion)
    print("\n __________ \n")
    print("Classification Report: \n __________ \n")
    print(classification_report(y_true, y_predicted))

    return y_true, y_predicted

y_real, y_predicted_yamnet = Confusion_Matrix(y_true, y_pred_yamnet, binarized_true = True, binarized_pred = True)
y_real, y_predicted_svm = Confusion_Matrix(y_true, y_pred_svm, binarized_true = True, binarized_pred = False, pred_prob=True)


# STACKING CLASSIFIER OF BOTH MODELS

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

y = y_real
# X = pd.DataFrame({'Yamnet':y_predicted_yamnet, 'SVM': y_predicted_svm})

'''
If we use the probabilities outputed by SVM, the stacking classifier works
much better than if we simply transform those outputs to [0,1] according to a threshold
'''

X = pd.DataFrame({'Yamnet':y_predicted_yamnet, 'SVM': pd.Series(y_pred_svm)})

estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42))) ]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf.fit(X_train, y_train).score(X_test, y_test) #  y_test == y_real.iloc[X_test.index]
y_pred_combined = clf.predict_proba(X_test)[:,1]  # The probability of getting the output as 1 (cough)
Confusion_Matrix(y_test, y_pred_combined, pred_prob=True)

y_pred_combined = clf.predict_proba(X)[:,1]
y_real, y_predicted_combined = Confusion_Matrix(y, y_pred_combined, pred_prob=True)

X_new = pd.DataFrame({'Yamnet':[0], 'SVM': [0.95]})
clf.predict_proba(X_new)[:,1]


# We verify that the optimal threshold outputs the best metrics

prueba = pd.Series(y_pred_combined)
prueba.loc[prueba >= 0.628] = 1
prueba.loc[prueba < 0.628] = 0
prueba = prueba.astype(int)
Confusion_Matrix(y, prueba)


# Import Joblib Module from Scikit Learn

import joblib

# Save RL_Model to file in the current working directory

joblib_file = "stacking_classifier.pkl"
joblib.dump(clf, joblib_file)

# Load from file

stacking_classifier = joblib.load(joblib_file)
stacking_classifier
stacking_classifier.predict_proba(X_new)[:,1]

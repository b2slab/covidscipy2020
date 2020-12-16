from sklearn import preprocessing
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/features_extracted.csv')
mid_feature_names = df.columns.values[0:-2]
data = df.to_numpy()
features = data[:, 0:-2]
labels = data[:, -2]
print(mid_feature_names)

# Normalization

data_scaled = preprocessing.StandardScaler().fit_transform(features)

ind = [1,103]

feature_names = mid_feature_names[ind]
cough_df = pd.DataFrame(data_scaled[:,ind], columns = feature_names)
cough_df["label"] = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
sns.pairplot(cough_df, hue="label")

# Using these two variables for a classification task,
# there is no hyperplane that can linearly separate
# the 2 classes. Then, we will probably need Kernels for SVM

'''
PCA to check how much redundant information is stored in the features.
'''

print("Explained variability")
pca = decomposition.PCA(whiten=False).fit(data_scaled)
print(100*pca.explained_variance_ratio_.cumsum())

data_reduced = decomposition.PCA(n_components=2,whiten=False).fit_transform(data_scaled)

df_pca = pd.DataFrame(data_reduced)
df_pca["label"] = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
sns.pairplot(df_pca, hue="label")

'''
Splitting into Test Data and Train Data
'''

lab = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
X_train, X_test, y_train, y_test = train_test_split(data_scaled, lab, test_size=0.30, random_state=50, shuffle=True)

'''
SVM Training with Grid Search
'''

def evaluation_SVM(tuned_parameters,X_train, y_train, X_test, y_test):
    scores = ['f1']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score,cv=3, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        print(clf.score(X_test,y_test))
    return clf.best_params_


from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn import decomposition
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

### Radial bases function kernel/Gaussian kernel

tuned_parameters_2 = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 75, 100, 200, 500, 750, 1000, 2000, 5000, 7000], 'kernel': ['rbf'], 'decision_function_shape': ['ovr'], 'gamma': [3, 2, 1, 1e-1, 1e-3, 1e-4, 1e-5],},]
rbfKernel_param=evaluation_SVM(tuned_parameters_2,X_train, y_train, X_test, y_test)

### Polynomial Kernel

tuned_parameters_3 = [ {'C': [0.01, 0.1, 1, 10, 25, 100], 'kernel': ['poly'], 'decision_function_shape': ['ovr'], 'degree': [1,2,3],'gamma': [5, 4, 3, 2, 1, 1e-1] },]
polyKernel_param=evaluation_SVM(tuned_parameters_3,X_train, y_train, X_test, y_test)

### Using the best polynomial kernel
rbfKernel_param = {'C': 7000, 'decision_function_shape': 'ovr', 'gamma': 1e-05, 'kernel': 'rbf'}
best_gaussian_model = SVC(**rbfKernel_param)
best_gaussian_model.fit(X_train, y_train)

### Storing classifier

import joblib
joblib.dump(best_gaussian_model, 'SVM_cough_classifier.joblib')

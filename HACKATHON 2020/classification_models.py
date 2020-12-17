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

# Check interactions between variables

ind = [1,103]

feature_names = mid_feature_names[ind]
cough_df = pd.DataFrame(features[:,ind], columns = feature_names)
cough_df["label"] = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
sns.pairplot(cough_df, hue="label")

# Using these two variables for a classification task,
# there is no hyperplane that can linearly separate
# the 2 classes. Then, we will probably need Kernels for SVM

'''
PCA to check how much redundant information is stored in the features.
'''

pca = decomposition.PCA(whiten=False).fit(features)
print(100*pca.explained_variance_ratio_.cumsum())

data_reduced = decomposition.PCA(n_components=2,whiten=False).fit_transform(features)

df_pca = pd.DataFrame(data_reduced)
df_pca["label"] = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
sns.pairplot(df_pca, hue="label")

'''
Splitting into Test Data and Train Data
'''

lab = pd.Series(labels).apply(lambda x: "cough" if x==1 else "no_cough")
X_train, X_test, y_train, y_test = train_test_split(features, lab, test_size=0.30, random_state=50, shuffle=True)

## Normalization --- standarize test set by the scaling used in train to avoid overfitting
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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


from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

### Radial bases function kernel/Gaussian kernel

tuned_parameters_1 = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 75, 100, 200, 500, 750, 1000, 2000, 5000, 7000], 'kernel': ['rbf'], 'decision_function_shape': ['ovr'], 'gamma': [3, 2, 1, 1e-1, 1e-3, 1e-4, 1e-5],},]
rbfKernel_param=evaluation_SVM(tuned_parameters_1,X_train, y_train, X_test, y_test)

### Polynomial Kernel

tuned_parameters_2 = [ {'C': [0.01, 0.1, 1, 10, 25, 100], 'kernel': ['poly'], 'decision_function_shape': ['ovr'], 'degree': [1,2,3],'gamma': [5, 4, 3, 2, 1, 1e-1] },]
polyKernel_param=evaluation_SVM(tuned_parameters_2,X_train, y_train, X_test, y_test)

### Using the best polynomial kernel
best_gaussian_model = SVC(**rbfKernel_param)
clf = best_gaussian_model.fit(X_train, y_train)

# Here we can visualize the decision boundary for the best Gaussian Model according to the grid search
def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    # plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, c = 'r')  # Plot the support vectors
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt

plt.figure()
plt.title("Gaussian SVC decision boundary", fontsize=16)
plot_decision_boundaries(X_train, y_train.apply(lambda x: 1 if x=='cough' else 0), SVC, **rbfKernel_param)
plt.show()

# Here we can visualize the decision boundary for the best Polynomial Model according to the grid search
best_polynomial_model = SVC(**polyKernel_param)
best_polynomial_model.fit(X_train, y_train)

plt.figure()
plt.title("Polynomial Kernel SVC decision boundary", fontsize=16)
plot_decision_boundaries(X_train, y_train.apply(lambda x: 1 if x=='cough' else 0), SVC, **polyKernel_param)
plt.show()


### Storing classifier
import joblib
joblib.dump(best_gaussian_model, 'SVM_cough_classifier.joblib')


'''
Train a classifier with PyAudioAnalysis
'''
from pyAudioAnalysis import audioTrainTest as aT
help(aT.extract_features_and_train)

cough_path = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/Cough/'
nocough_path = 'C:/Users/Guillem/Desktop/HACKATHON 2020/Unlabeled audio/TRAIN/No_Cough/'
svm_linear = aT.extract_features_and_train([cough_path,nocough_path], 0.2, 0.2, aT.shortTermWindow, aT.shortTermStep, "svm", "svm_linear", False, train_percentage=0.80)

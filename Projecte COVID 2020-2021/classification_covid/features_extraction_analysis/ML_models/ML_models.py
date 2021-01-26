import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/audios/features_extracted.xlsx', index_col=0)
df.shape

metadata = pd.read_excel('C:/Users/Guillem/Desktop/HACKATHON 2020/Labeled audio/metadata.xlsx')
metadata.columns
metadata = metadata[['patient_id', 'age', 'gender', 'asthma', 'cough', 'smoker', 'hypertension', 'cold',
       'diabetes', 'ihd', 'bd', 'st', 'fever', 'ftg', 'mp', 'loss_of_smell',
       'pneumonia', 'diarrhoea', 'cld']]

metadata = pd.concat([metadata.iloc[:,:2], pd.get_dummies(metadata.iloc[:,2:], sparse=True)], axis = 1)
metadata.head()
metadata = metadata.rename(columns = {'patient_id':'filename'})
df['filename'] = df['filename'].map(lambda name: name.split('_')[0])

df = df.merge(metadata, on = ['filename'])
df.to_excel('features_metadata.xlsx')
df = df.drop('filename', axis=1)

# splitting by class
pos = df[df.label == 1]
neg = df[df.label == 0]


from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, shuffle = True)

lzyclf = LazyClassifier(verbose=0, ignore_warnings = True, custom_metric=None)
models, predictions = lzyclf.fit(X_train, X_test, y_train, y_test)

models
predictions

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,10))
ax = sns.barplot(y = models.index, x = 'ROC AUC', data = models)



'''
Nearest Centroid
'''

##### ADD CLASS WEIGHTS TO model.score(x,y, sample_weights)

'''
neg = 1616 + 693
pos = 638 + 274
total = pos + neg

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
'''

n_samples = len(y)
n_classes = len(y.unique())
n_samples0 = y.value_counts()[0]
n_samples1 = y.value_counts()[1]

w0 = n_samples / (n_classes * n_samples0)
w1 = n_samples / (n_classes * n_samples1)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

weights = y_train.map(lambda y: w0 if y == 0 else w1)


from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report
# Creating the Nearest Centroid Clissifier
model = NearestCentroid()

# Training the classifier
model.fit(X_train, y_train.values.ravel())

model.score(X_train, y_train, sample_weight=weights)

# Printing Accuracy on Training and Test sets
print(f"Training Set Score : {model.score(X_train, y_train) * 100} %")
print(f"Test Set Score : {model.score(X_test, y_test) * 100} %")

# Printing classification report of classifier on the test set set data
print(f"Model Classification Report : \n{classification_report(y_test, model.predict(X_test))}")



'''
Extra Tree Classifier
'''

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


model = ExtraTreesClassifier(random_state=42, class_weight='balanced')
cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
print("The average Cross validation score is ", round(cv_score*100,2))

'''
GRID SEARCH CV
'''
param_grid = {'n_estimators': [50, 100, 250, 500, 750, 1000],
              'max_depth': [5, 10, 50],
              'criterion': ['gini', 'entropy'],
              'bootstrap':[False,True],
              'max_features':['auto', 'sqrt', 'log2']}

clf = GridSearchCV(model, param_grid, n_jobs=6, cv=5, verbose=3)
clf.fit(X_train, y_train)

clf.best_params_
clf.best_score_


pred = clf.predict_proba(X_new)[:,1]
pred  #### Probability of being Positive

Confusion_Matrix(y_true = y_new, y_predicted = pred, pred_prob=True)


### Concatenate X_test + X_new
pred = clf.predict_proba(pd.concat([X_test, X_new], axis=0))[:,1]
pred.shape

Confusion_Matrix(np.array(pd.concat([y_test, y_new], axis = 0)), pred, pred_prob=True)

'''
RANDOMIZED SEARCH CV
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 250, stop = 750, num = 20)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,5,10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [False]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model = ExtraTreesClassifier(random_state=42, class_weight='balanced')
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
#kfold = KFold(n_splits=5, shuffle=True, random_state=42)
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=1, random_state=42, n_jobs = 6)
rf_random.fit(X_train, y_train)
rf_random.best_params_
rf_random.best_score_

etc_pred = rf_random.predict_proba(X_new)[:,1]
Confusion_Matrix(y_new, etc_pred, pred_prob=True)

Confusion_Matrix(np.array(pd.concat([y_test, y_new], axis = 0)), rf_random.predict_proba(pd.concat([X_test, X_new], axis=0))[:,1], pred_prob=True)



'''
Bayesian OPTIMIZATION
'''

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# log-uniform: understand as search over p = exp(x) by varying x
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
        ExtraTreesClassifier(random_state=42, class_weight='balanced'),
        {
         'n_estimators': Integer(200,600, prior = 'log-uniform'),
         'max_features': Categorical(['auto']),
         'max_depth': Integer(5,25),
         'min_samples_split': Integer(2,8),
         'min_samples_leaf': Integer(2,6),
         'bootstrap': Categorical([False]),
        },
        n_iter=300,
        random_state=42,
        cv = 5,
        n_jobs=6,
        verbose = 1
        )

# y_metales.values.reshape(-1,1)
# Fit the random search model
opt.fit(X_train, y_train)
opt.best_params_
opt.best_score_

opt_pred = opt.predict_proba(X_new)[:,1]
Confusion_Matrix(y_new, opt_pred, pred_prob=True)


Confusion_Matrix(np.array(pd.concat([y_test, y_new], axis = 0)), opt.predict_proba(pd.concat([X_test, X_new], axis=0))[:,1], pred_prob=bool)



'''
Best parameters
'''

best_parameters = {'GridSearchCV': {'bootstrap': False,
                                    'criterion': 'gini',
                                    'max_depth': 10,
                                    'max_features': 'auto',
                                    'n_estimators': 500},
                    'RandomizedSearchCV': {'n_estimators': 276,
                                           'min_samples_split': 5,
                                           'min_samples_leaf': 2,
                                           'max_features': 'auto',
                                           'max_depth': 18,
                                           'bootstrap': False},
                    'BayesSearchCV': {'n_estimators': 363,
                                      'min_samples_split': 5,
                                      'min_samples_leaf': 2,
                                      'max_features': 'auto',
                                      'max_depth': 14,
                                      'bootstrap': False}}


clf.best_params_
clf.best_score_
rf_random.best_params_
opt.best_params_

'''
SHAP VALUES
'''

import shap  # package used to calculate Shap values

model = ExtraTreesClassifier(random_state=42, verbose = 1, class_weight='balanced', n_jobs=6,
                             bootstrap=False, criterion='gini',max_depth=10,max_features='auto',n_estimators=500)

model = model.fit(X_train,y_train)
feature_importance = model.feature_importances_

# Normalizing the individual importances
feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                        model.estimators_],
                                        axis = 0)

f_imp = pd.DataFrame({'column': X_train.columns[np.where(feature_importance_normalized>0.013)],
                      'feature_imp':feature_importance_normalized[np.where(feature_importance_normalized>0.013)]})


# Plotting a Bar Graph to compare the models
plt.barh(f_imp['column'].iloc[f_imp['feature_imp'].sort_values().index], f_imp['feature_imp'].sort_values())
plt.xlabel('Feature Labels')
plt.title('Comparison of different Feature Importances')
plt.show()

# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate Shap values
shap_values = explainer.shap_values(X_new)
np.shape(shap_values)

# summarize the effects of all the features
# load JS visualization code to notebook
shap.initjs()
shap.summary_plot(np.array(shap_values)[1,:,:], X_new)



plt.scatter(X_new['spectral_spread_mean'], np.array(shap_values)[1,:,X_new.columns.get_loc('spectral_spread_mean')])


# create a dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("spectral_spread_mean", np.array(shap_values)[1,:,:], X_new)

'''
LGBMClassifier
'''

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

LGBM = LGBMClassifier(random_state=42, class_weight='balanced')
cv_score = cross_val_score(LGBM, X_train, y_train, cv=5).mean()
print("The average Cross validation score is {}%".format(round(cv_score*100,2)))

rs_params = {
        'bagging_fraction': (0.5, 0.8),
        'bagging_frequency': (5, 8),

        'feature_fraction': (0.5, 0.8),
        'max_depth': (10, 13),
        'min_data_in_leaf': (90, 120),
        'num_leaves': (1200, 1550)
}

rs_cv.best_params_

# Initialize a RandomizedSearchCV object using 5-fold CV-
rs_cv = RandomizedSearchCV(estimator=LGBM, param_distributions=rs_params, cv = 5, n_iter=100,verbose=1, n_jobs=6)

# Train on training data-
rs_cv.fit(X_train, y_train)

pred_lgbm = rs_cv.predict_proba(X_new)[:,1]

Confusion_Matrix(y_true = y_new, y_predicted = pred_lgbm, pred_prob=True)


'''
Save the models
'''

import joblib

# Save RL_Model to file in the current working directory

joblib_file = "extratree_classifier.pkl"
joblib.dump(clf, joblib_file)

joblib_file = "lgbm_classifier.pkl"
joblib.dump(rs_cv, joblib_file)


'''
Decision Tree Classifier


from sklearn.tree import DecisionTreeClassifier


model2 = DecisionTreeClassifier(random_state=42, class_weight='balanced')
cv_score = cross_val_score(model2, X_train, y_train, cv=5).mean()
print("The average Cross validation score is ", round(cv_score*100,2))

param_grid = {#'n_estimators': [50, 100, 250, 500, 750, 1000],
              'max_depth': range(1,20,2),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              #'bootstrap':[False,True],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_split' : range(10,500,20)}

clf2 = GridSearchCV(model2, param_grid, n_jobs=6, cv=5, verbose=3)
clf2.fit(X_train, y_train)
clf2.best_score_
'''


'''
Evaluation of the model in New Coswara Data
'''

df_new = pd.read_excel('C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/New Coswara data/features_extracted.xlsx', index_col=0)
df_new['filename'] = df_new['filename'].map(lambda name: name.split('_')[0])
df_new = df_new.merge(metadata, on = ['filename'])
df_new.to_excel('newdata_extracted_metadata.xlsx')


df_new = df_new.drop('filename', axis=1)

X_new = df_new.drop('label', axis = 1)
y_new = df_new['label']


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

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

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




'''
Stacking classifier
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

estimators = [
     ('etc', ExtraTreesClassifier(random_state=42, class_weight='balanced')),
     ('lgbm', LGBMClassifier(random_state=42, class_weight='balanced')) ]

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, class_weight='balanced'))

rs_cv.best_params_
clf.best_params_

params = {'etc__bootstrap': [False],
          'etc__criterion': ['gini'],
          'etc__max_depth': [10],
          'etc__max_features': ['auto'],
          'etc__n_estimators': [100,500],
          'lgbm__num_leaves': [1200],
          'lgbm__min_data_in_leaf': [90],
          'lgbm__max_depth': [10],
          'lgbm__feature_fraction': [0.5],
          'lgbm__bagging_frequency': [5],
          'lgbm__bagging_fraction': [0.5]}

gridcv = GridSearchCV(stack, params, verbose = 1, n_jobs=6)
final_model = gridcv.fit(X_train, y_train)

y_pred = final_model.predict_proba(X_new)[:,1]
Confusion_Matrix(y_new, y_pred, pred_prob=True)

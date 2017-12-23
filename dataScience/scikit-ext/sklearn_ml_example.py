import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# Import the tools to perform cross-validation.
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Evaluate model performance.
from sklearn.metrics import mean_squared_error, r2_score

# Persist out model forr future use.
from sklearn.externals import joblib

# Load red wine data.
# read_csv() : we can load any CSV file, even from a remote URL!
dataset_dir = './winequality-red.csv'
data = pd.read_csv(dataset_dir, sep = ';')

# Output the first 5 rows of data.
print data.head()
print data.shape
print '\n' + '#' * 80 + '\n'

# Print the summary statistics.
print data.describe()
print '\n' + '#' * 80 + '\n'

# Split data into training and test sets.
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 123,
                                                    stratify = y)

# Data preprocessing steps.

## The code we won't use ##
#X_train_scaled = preprocessing.scale(X_train)
#print X_train_scaled
#print '\n' + '#' * 80 + '\n'
#print X_train_scaled.mean(axis = 0)
#print X_train_scaled.std(axis = 0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print X_train_scaled.mean(axis = 0)
print X_train_scaled.std(axis = 0)
print '\n' + '#' * 80 + '\n'

# Transform test set using the means from the traning set.
X_test_scaled = scaler.transform(X_test)
print X_test_scaled.mean(axis = 0)
print X_test_scaled.std(axis = 0)

# Pipeline with preprocessing and model.
pineline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators = 100))

# List tunable hyperparameters
print pineline.get_params()
# ...
# 'randomforestregressor__criterion': 'mse',
# 'randomforestregressor__max_depth': None,
# 'randomforestregressor__max_features': 'auto',
# 'randomforestregressor__max_leaf_nodes': None,
# ...

# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Cross-validation
clf = GridSearchCV(pineline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)

print clf.best_params_

# Refit on the entire training set.(On by default)
print clf.refit

# Evaluate the model pineline on test data.
y_pred = clf.predict(X_test)
print r2_score(y_test, y_pred)
print mean_squared_error(y_test, y_pred)

# Save the model for future use.
joblib.dump(clf, 'rf_regressor.pkl')

# Load model from .pkl file
clf2 = joblib.load('rf_regressor.pkl')
clf2.predict(X_test)
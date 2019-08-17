# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('S1R1.csv')
X = dataset.iloc[:, 0:61].values
y = dataset.iloc[:, 92:95].values
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:,:] = imputer.transform(X[:, :])
imputer = imputer.fit(y[:, :])
y[:,:] = imputer.transform(y[:, :])
#Loading the model
#from sklearn.externals import joblib
#regressor = joblib.load('ANN_s12_1_R')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages


import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
regressor = Sequential()
# Adding the input layer and the first hidden layer
#regressor.add(Dense())
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
# Adding the second hidden layer
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
regressor.add(Dense(units = 48, kernel_initializer = 'glorot_uniform', activation = 'linear'))
# Adding the output layer
regressor.add(Dense(units = 3, kernel_initializer = 'glorot_uniform', activation = 'linear'))
# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 100, epochs = 5000)

y_pred = regressor.predict(X_test)
#Saving the model
from sklearn.externals import joblib
joblib.dump(regressor,'ANN_s1_1_R')
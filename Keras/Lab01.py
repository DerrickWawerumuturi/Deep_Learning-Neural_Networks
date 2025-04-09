## 1. Import required libraries
import pandas as pd
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.layers import Input
from keras.src.callbacks import EarlyStopping
from keras.api.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error

## 2. Load dataset
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

## 3. Split the data to predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# normalize data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1]

## 3. Build a neural network

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# fit and train the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=200, verbose=2)
predictions = model.predict(predictors_norm)

print(predictions[:5])
mse = mean_squared_error(target, predictions)
mae = mean_absolute_error(target, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

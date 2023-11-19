import keras.regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Here we go through how to select the right neural network layer structure.

# some random input values
x_original = np.arange(-20000, 20000, 100)
y_original = np.arange(-200, 200, 1)

# Make a 2d array so that it is easy to call other functions.
x = np.expand_dims(x_original, axis=1)
y = np.expand_dims(y_original, axis=1)

# Split the data test in to training and test data.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Feature scaling - refer to feature-scaling.py for more details
scaler = StandardScaler()
x_train_mapped_scaled = scaler.fit_transform(x_train)
x_cv_mapped_scaled = scaler.transform(x_cv)
x_test_mapped_scaled = scaler.transform(x_test)

# Build a model. Start of a with 2 hidden layers.
model = Sequential(
    [
        Dense(3, activation='linear', kernel_regularizer=keras.regularizers.L2(0.01)),
        Dense(2, activation='linear', kernel_regularizer=keras.regularizers.L2(0.01)),
        Dense(1, activation='linear', kernel_regularizer=keras.regularizers.L2(0.01))
    ]
)

# We are using mean squared error as this is a continuous prediction.
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1),
)

model.fit(
    x_train_mapped_scaled, y_train,
    epochs=300,
    verbose=0,
    batch_size=64  # This is optional. batch_size will default to 32 if not specified.
)

# Let's see what the model predicts for the training set.
y_hat = model.predict(x_train_mapped_scaled)

# What is the mean squared error for training set. This gives an idea of how well the model has fit the training data.
train_mse = mean_squared_error(y_train, y_hat)

# Prediction for cv

y_hat_cv = model.predict(x_cv_mapped_scaled);

# What is it for new values.
cv_mse = mean_squared_error(y_cv, y_hat_cv)

print("Training data mean squared error", train_mse)
print("Cross validation data mean squared error", cv_mse)

# Now we change the model parameters, like add more hidden layers or increase the number of neurons per layer or if
# it has variance then add regularization and then calculate MSE on training and cv data Test data is only to know
# how the model does on unknown and opinionated data. Remember CV data helps us chose the model structure. Test data
# is used to calculate generalisation error.


# Checking how it predicts
print(model.predict(scaler.transform([[20000]])))

# This uses the class API of Sequential model. There is also a functional API which can be found here -
# https://keras.io/guides/functional_api/

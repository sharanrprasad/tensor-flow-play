import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Gives ana array from -20 to +20 with intervals of 1.
X1 = np.arange(-20, 20, 1)

X2 = (X1 ** 2)

X2_normal = X2 / np.max(X2)

X3 = (X1 ** 3)

X3_normal = X3/np.max(X3)

# Let's just say this the required output =
Y = X1 ** 2 + 24

# Stack 1-D arrays as columns into a 2-D array.
X = np.c_[X1, X2_normal, X3_normal]
X_train = X[0:30]
X_test = X[30:40]

Y_train = Y[0:30]
Y_test = Y[30:40]




model = Sequential(
    [
        Dense(1, activation='linear'),
    ]
)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(x=X_train,y=Y_train, epochs=200)

print(model.predict(X_test))

print(Y_test)
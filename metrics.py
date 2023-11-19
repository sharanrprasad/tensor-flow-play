from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime


# Create a dummy data object.
data = np.random.randint(0, 256, size=(72, 200)).astype("float32")

labels = np.random.randint(0, 256, size=(72,)).astype('float32')

# Metrics are defined at compile functions

inputs = keras.Input(shape=(200,), batch_size=32)
x = keras.layers.Dense(64, activation='relu')(inputs)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

early_stop_cb = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=5, verbose=1, mode='auto')

# Reduce the learning rate when the loss kind of plateaus.
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.05, patience=2, verbose=1, min_delta=10, mode='min')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.1), loss='mse',
              metrics=[keras.metrics.mean_squared_error,
                       keras.metrics.mean_absolute_error,
                       keras.metrics.mean_absolute_percentage_error,
                       ])

# Uses 20% of the data for validation.
history = model.fit(data, labels, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stop_cb, reduce_lr])


figure, axis = plt.subplots()
axis.plot(history.history['mean_squared_error'])
axis.plot(history.history['val_mean_squared_error'])
axis.set_ylabel('MSE')
axis.set_xlabel('Epoch')
axis.legend(['Train', 'Validation'], loc='upper right')
plt.show()
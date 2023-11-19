from tensorflow import keras
import numpy as np

# Functional API offers more control compared to the class API used in neural_networks_selection.py More info here -
# https://keras.io/getting_started/intro_to_keras_for_engineers/#building-models-with-the-keras-functional-api

# In functional API there will be an Input layer -

# This is an input layer with each input being 200 dimensional vector. Batch size is not specified in shape that
# comes as it's own arg which is also optional. If specified, then all the layers will also reflect that. This can be
# overriden while using predict (Need to figure out what exactly this does) and fit methods.
inputs = keras.Input(shape=(200,), batch_size=32)

# After defining your input(s), you can chain layer transformations on top of your inputs, until your final output

# Note that dense here is just a layer. We can add other types of layers as well.
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dense(16, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

# Once you have defined the directed acyclic graph of layers that turns your input(s) into your outputs, instantiate
# a Model object
model = keras.Model(inputs=inputs, outputs=outputs)

# Create a dummy data object.
data = np.random.randint(0, 256, size=(72, 200)).astype("float32")

labels = np.random.randint(0, 256, size=(72,)).astype('float32')
processed_data = model(data)

# This would be (72, 1).
print(processed_data.shape)

print(model.summary())

# Next step is provide the model with optimiser and error function using compile function. This needs to happen
# before calling model.fit().

model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.1), loss='mse',
              metrics='sparse_categorical_accuracy')

model.fit(data, labels, batch_size=64, epochs=1)

# If not all data can be fit in to CPU/RAM at the same time, batch_size can help here.
print(model.predict(np.random.randint(0, 256, size=(10, 200)).astype("float32"), batch_size=2))

# Example creating a Content based filtering (Recommendation) algorithm using Functional API


# user_NN = tf.keras.models.Sequential([
#     ### START CODE HERE ###
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(64, activation='relu')
#
#
#     ### END CODE HERE ###
# ])
#
# item_NN = tf.keras.models.Sequential([
#     ### START CODE HERE ###
#     keras.layers.Dense(64, activation='relu')
#     keras.layers.Dense(64, activation='relu')
#     ### END CODE HERE ###
# ])
#
# # create the user input and point to the base network
# input_user = tf.keras.layers.Input(shape=(num_user_features))
# vu = user_NN(input_user)
# vu = tf.linalg.l2_normalize(vu, axis=1)
#
# # create the item input and point to the base network
# input_item = tf.keras.layers.Input(shape=(num_item_features))
# vm = item_NN(input_item)
# vm = tf.linalg.l2_normalize(vm, axis=1)
#
# # compute the dot product of the two vectors vu and vm
# output = tf.keras.layers.Dot(axes=1)([vu, vm])
# specify the inputs and output of the model
# model = tf.keras.Model([input_user, input_item], output)
#
# model.summary()

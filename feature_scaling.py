import numpy as np
from sklearn.preprocessing import StandardScaler

# Feature scaling is achieved by formula z = (x - 𝜇)/𝜎
# where  𝜇 is the mean of the feature values and  𝜎 is the standard deviation. This process is called Standardisation.
# This can be achieved in scikit learn as follows -

x = np.array([2, 4, 8, 16, 32, 64])

# Convert 1-D arrays into 2-D because the commands later will require it
x_two = np.expand_dims(x, axis=1)

scaler_linear = StandardScaler()
x_scaled = scaler_linear.fit_transform(x_two)

print(x_scaled)

# An important thing to note when scaling features of a cross validation set or a test set is that - We should use
# the mean and standard deviation of the training set. Say that your training set has an input feature equal to 500
# which is scaled down to 0.5 using the z-score. After training, your model is able to accurately map this scaled
# input x=0.5 to the target output y=300. Now let's say that you deployed this model and one of your users fed it a
# sample equal to 500. If you get this input sample's z-score using any other values of the mean and standard
# deviation, then it might not be scaled to 0.5 and your model will most likely make a wrong prediction (i.e. not
# equal to y=300).


# How to do that is -
x_cv = np.array([12, 13])

x_cv_two = np.expand_dims(x_cv, axis=1)

# Now just use the scalar from before
x_cv_scaled = scaler_linear.transform(x_cv_two)

print(x_cv_scaled)

# Standardisation mentioned above is a type of scaling data. The above equation gives us a vector which has a mean of
# zero and a standard deviation of 1 and also the vector is a normal distribution(bell curve).
# We can generate normal distribution from a given data by also keeping the same mean and stanadard deviation.
# More on that in recommender-system collaborative-learning.


# Vector normalisation is also another type used to scale data. There are different ways we can normalise a
# vector, like 'euclidian'. Refer to tensors.py for more info on normalisation.

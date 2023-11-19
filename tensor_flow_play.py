import pandas
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pandas.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scaling features is preferred as it gives a friendlier loss values and learning rates.
# Even though scaling labels is not essential, scaling features is.
training_df["median_house_value"] /= 1000

# gives out the mean, standard deviation and quantile(divided into four parts)
# print(training_df.describe())  # print this

# gives out the correlation between different features
# print(training_df.corr())




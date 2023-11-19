import numpy as np
import pandas as pd

# Categorical data is a way to represent data that can be divided into specific categories or groups.
# For example, if you have a dataset of people and their ages,
# you could use categories to group the ages into ranges such as "child", "teenager", "adult" and "senior"

# Category can be defined using the series api itself. Just change the d-type.
# This creates one category for each value.
gender = pd.Series(["male", "female", "non-binary"], dtype="category")

print(gender)

ages = pd.Series([25, 35, 42, 18, 55, 27, 31, 20, 49])

# Series can be converted to a category
# This also creates one category for each value.
ages_C = pd.Categorical(ages)


# We can also specify what values actually constitutes a category and what is not.
# Here only 25,35 and 42 becomes category and rest is NAN.
ages_Check = pd.Categorical(ages, [25, 35, 42])


# Use case. Category can be used within a pandas data frame. Say we have the below data frame
df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B'],
                   'value': [10, 20, 30, 40, 50]})

# convert the 'category' column to a Categorical data type
df["category"] = pd.Categorical(df["category"])

# group the data by category and calculate the mean value for each group
grouped = df.groupby('category')['value'].mean()


# encode the 'category' column using one-hot encoding. converts to a dataframe with columns
# like  category_A  category_B
encoded = pd.get_dummies(df['category'], prefix='category')
# print(encoded)

# Convert the latitude data house price median to categorical data
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

resolution_in_degrees = 1.0

# Create a list of numbers representing the bucket boundaries for latitude.
latitude_boundaries = list(np.arange(int(min(test_df['latitude'])),
                                     int(max(test_df['latitude'])),
                                     resolution_in_degrees))


# Cut method returns a category as well.
test_df['latitude_category'] = pd.cut(test_df['latitude'], latitude_boundaries)

print(test_df['latitude_category'].head())

# One-hot encode the latitude categories. This returns a data frame
one_hot_latitude = pd.get_dummies(test_df['latitude_category'])

# print(pd.concat([test_df, one_hot_latitude], axis=1).head())


# Doing a feature cross between latitude and longitude one hot encoded values

# Create a list of numbers representing the bucket boundaries for longitude.
longitude_boundaries = list(np.arange(int(min(test_df['longitude'])),
                                     int(max(test_df['longitude'])),
                                     resolution_in_degrees))

test_df["longitude_category"] = pd.cut(test_df["longitude"], longitude_boundaries)

one_hot_longitude = pd.get_dummies(test_df['longitude_category'])

result = pd.DataFrame()
for latitude_name, latitude_value in one_hot_latitude.items():
    for longitude_name, longitude_value in one_hot_longitude.items():
        feature_cross_name = latitude_name.__str__() + "_" + longitude_name.__str__()
        feature_cross_value = latitude_value * longitude_value
        result[feature_cross_name] = feature_cross_value

# print(result.head())






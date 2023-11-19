import numpy as np
import pandas
import statsmodels.formula.api as smf
import graphing
import plotly.express

# Make a dictionary of data for boot sizes
# and harness size in cm


dataset = pandas.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv')

data_smaller_paws = dataset[dataset.boot_size < 40].copy()

# Print information about this
print(f"We now have {len(data_smaller_paws)} rows in our dataset. The last few rows are:")
print(data_smaller_paws.tail())

plotly.express.scatter(data_smaller_paws, x="harness_size", y="boot_size").show()


# formula = "boot_size ~ harness_size"
#
# model = smf.ols(formula = formula, data = dataset)
#
# fitted_model = model.fit()
#
#
# harness_size = { 'harness_size' : [49] }
#
# # Use the model to predict what size of boots the dog will fit
# approximate_boot_size = fitted_model.predict(harness_size)
#
# # Print the result
# print("Estimated approximate_boot_size:")
# print(approximate_boot_size[0])
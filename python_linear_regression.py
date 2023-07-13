#!/usr/bin/env python
# # Python Linear Regression

# ## Import matplotlib, pandas, sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os
print()


print ("Running Linear Modeling of Python Script\n")


# ##(Import file)
if len(sys.argv) <2:
    print("File unreadable/File too short")
    syn.exit (-1)
        
if len(sys.argv) >2:
    print("File working")
    
filename = sys.argv[1]
print()
print ("Loading filename {}".format(filename))
print()

# ## Load File
dataset = pd.read_csv(filename)
print (dataset)
print()

# # Fitting Linear Regression to the Dataset
print (dataset.describe())
model = LinearRegression ()
model.fit(dataset[['x']],dataset[['y']])

# ## General Statistics
score = model.score(dataset[['x']], dataset[['y']])
print()
print("Model Score:", score)


# ## Visualizing the Linear Regression Results

# ### Scatter Plot of Dataset
plt.scatter(dataset[['x']],dataset[['y']], color = 'blue')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("scatter_plot.png")



# ### Regression Line with Data Points
plt.scatter(dataset[['x']],dataset[['y']], color = 'blue')
plt.plot(dataset[['x']],model.predict(dataset[['x']]),color = 'red')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("regression_line_with_datapoints.png")



print("Thank you for using Noah's script:)" )
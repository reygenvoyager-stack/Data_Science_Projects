#simple linear regression based on discount amount using retail_sales_dataset Excel

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import seaborn as sns
import numpy as np

retail_sales = pd.read_excel("/Users/sandytran/Desktop/retail_sales_dataset.xlsx", sheet_name="Transactions")
retail_sales = pd.DataFrame(retail_sales)

#split dataset to X and Y variables
#x variable is 'Discount'
X = pd.DataFrame(retail_sales.Discount)
Y = pd.DataFrame(retail_sales.Quantity)

#Perform 80/20 Data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#data dimension
X_train.shape, Y_train.shape
X_test.shape, Y_test.shape

#create linear regression model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#apply trained model to make predictions on test set
Y_pred = model.predict(X_test)

#print model performance
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

#scatter plot
np.array(Y_test)
plt.scatter(Y_test, Y_pred)
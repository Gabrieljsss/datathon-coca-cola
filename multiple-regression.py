import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('monthly-dataset.csv', sep=',')

def format_sale_string(value):
    '''
    Convert sales number string to floating point
    '''
    value = value.replace('.', '')
    value = value.replace(',', '.')
    return float(value)

X = dataset[['Month','PMC']]
y = dataset[['sum_Total']]

X['PMC'] = X['PMC'].apply(format_sale_string)

y['sum_Total'] = y['sum_Total'].apply(format_sale_string)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
#plt.scatter(y_test, y_pred,  color='black')
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='black') #Residual plot

plt.xticks(())
plt.yticks(())

plt.show()
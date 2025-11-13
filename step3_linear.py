#Linear Analysis_One parameters (in fact it is the same as the above "two parameters").
print('''\
------------------ ^^ ------------------
Step 3 for machine learning.
This program was written by Dawei Wen (温大尉)(*^^*).
If you have any question, please contact:
E-mail: ontaii@163.com.
Google Scholar: https://scholar.google.com/citations?hl=ja&user=U13L9sEAAAAJ
Research Gate: https://www.researchgate.net/profile/Dawei-Wen-2
------------------ ^_^ ------------------\
''')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

Matrix = input('Enter the file name of the matrix with extension for machine learning, for example, Matrix.xlsx:\n')
Matrix = pd.read_excel(Matrix)


X = Matrix[['LIR']] # Luminescence Intensity Ratio.
y = Matrix['Pressure (GPa)']

# Scaling
#scaler = StandardScaler()
#X_Scaled = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

'''Method'''
LR_RidgeCV = linear_model.RidgeCV(cv=5, alphas=(1.e-10,1,10))

'''Fitting'''
Method = LR_RidgeCV  # LR/LR_RidgeCV/SGDR/KNR/RNR/KRR/SVMR/XGB/MLPR
Method.fit(X, y)
y_pred = Method.predict(X)

# Evaluation2
print('Method:', Method)
print("Mean squared error: %.4f" % mean_squared_error(y, y_pred))  # The mean squared error
print("Mean absolute error: %.4f" % mean_absolute_error(y, y_pred))  # The mean squared error
print("Coefficient of determination: %.4f" % r2_score(y, y_pred))  # The coefficient of determination: 1 is perfect prediction
for i, j in zip(y_pred, y):
    print(round(float(i), 4), 'vs', j, '(GPa)')

#For linear model only!!!!!!!
print('Coefficients (2) and Intercept (1):')
print(Method.coef_,Method.intercept_)
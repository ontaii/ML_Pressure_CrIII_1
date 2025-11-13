#Linear Analysis_One parameter: pressure vs peak wavelength of 2E in Cr3+
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model, neighbors, svm
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
#import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

Matrix = input('Enter the file name of the matrix with extension for machine learning, for example, Matrix.xlsx:\n')
Matrix = pd.read_excel(Matrix)


X = Matrix[['Peak (nm)']] # x

y = Matrix['Pressure (GPa)'] #y

# Scaling
#scaler = StandardScaler()
#X_Scaled = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

'''Methods'''
LR = linear_model.LinearRegression()  # LinearRegression
LR_RidgeCV = linear_model.RidgeCV(cv=5, alphas=(1.e-10,1,10))
KNR = neighbors.KNeighborsRegressor(weights='distance',n_neighbors=2,algorithm='kd_tree') 
# weights can be "uniform"/"distance"，algorithm can be 'auto', 'ball_tree', 'kd_tree', 'brute',default:metric='minkowski'
RNR = neighbors.RadiusNeighborsRegressor(weights='distance')
KRR = KernelRidge(
    alpha=0.01,
    kernel='poly',
    gamma=None,
    degree=5,
    coef0=0.01
)
SVMR = svm.SVR(kernel='rbf', tol=0.00001)  # No coefficients output
'''
XGB = xgb.XGBRegressor(
    objective='reg:squarederror',  # regression task
    n_estimators=50,  # number of trees
    max_depth=5,  # maximum depth of trees
    learning_rate=0.01,
    #gamma=0.3,
    reg_lambda=0.5,
    #reg_alpha=0.5
)
'''
MLPR = MLPRegressor(
    hidden_layer_sizes=(3,),
    activation='relu',
    solver='adam',
    max_iter=30000,
    random_state=15,
)

'''Fitting'''
Method = LR_RidgeCV  # LR/LR_RidgeCV/SGDR/KNR/RNR/KRR/SVMR/XGB/MLPR
Method.fit(X, y)
y_pred = Method.predict(X)

''' 
# Evaluation
print('Method:', Method)
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))  # The mean squared error
print("Mean absolute error: %.2f" % mean_absolute_error(y, y_pred))  # The mean squared error
print("Coefficient of determination: %.2f" % r2_score(y, y_pred))  # The coefficient of determination: 1 is perfect prediction
for i, j in zip(y_pred, range(0, 11)):
    print(round(float(i), 4), 'vs', j, '(GPa)')
'''

# Evaluation2
print('Method:', Method)
print("Mean squared error: %.4f" % mean_squared_error(y, y_pred))  # The mean squared error
print("Mean absolute error: %.4f" % mean_absolute_error(y, y_pred))  # The mean squared error
print("Coefficient of determination: %.4f" % r2_score(y, y_pred))  # The coefficient of determination: 1 is perfect prediction
for i, j in zip(y_pred, y):
    print(round(float(i), 4), 'vs', j, '(GPa)')

'''
print('---------')
y_pred_test = Method.predict(X_test)
for i, j in zip(y_pred_test, y_test):
    print(round(float(i), 4), 'vs', j, '(GPa)')
'''
#For linear model only!!!!!!!
print('Coefficients (2) and Intercept (1):')
print(Method.coef_,Method.intercept_)
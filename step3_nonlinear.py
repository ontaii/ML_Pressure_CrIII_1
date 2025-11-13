#Non-Linear Analysis
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

Matrix = input('Enter the file name of the matrix with extension for machine learning, for example, Matrix_YAl.xlsx:\n')
Matrix = pd.read_excel(Matrix)


X = Matrix[['LIR']]
y = Matrix['Pressure (GPa)']

# Scaling
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.1, random_state=0)

'''Methods'''
LR = linear_model.LinearRegression()  # LinearRegression
LR_RidgeCV = linear_model.RidgeCV(cv=5, alphas=np.logspace(-10, 0, 11))
KNR = neighbors.KNeighborsRegressor(weights='distance',n_neighbors=2,algorithm='kd_tree')
# weights can be "uniform"/"distance"，algorithm can be 'auto', 'ball_tree', 'kd_tree', 'brute',default:metric='minkowski'
RNR = neighbors.RadiusNeighborsRegressor(weights='distance')
KRR = KernelRidge(
    alpha=0.001,          # 正则化强度（越大，模型越简单）
    kernel='rbf',       # 核函数：'rbf'（高斯核）、'poly'、'linear'等
    gamma=0.5,         # 核系数（默认1/n_features，或可手动设置如0.1）
    #degree=3,           # 多项式核的阶数（仅当kernel='poly'时有效）
    coef0=1             # 核函数中的常数项（对'poly'和'sigmoid'核有效）
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
    hidden_layer_sizes=(3,),  # hidden_layer_sizes=(4,3)2个隐藏层
    activation='relu',  # 激活函数
    solver='adam',  # 优化算法
    max_iter=30000,  # 最大迭代次数
    random_state=15,
)

'''Fitting'''
Method = KRR  # LR/LR_RidgeCV/SGDR/KNR/RNR/KRR/KRR_poly/SVMR/XGB/MLPR
Method.fit(X_Scaled, y)
y_pred = Method.predict(X_Scaled)

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

print('---------')
y_pred_test = Method.predict(X_Scaled)
for i, j in zip(y_pred, y):
    print(round(float(i), 4), 'vs', j, '(GPa)')
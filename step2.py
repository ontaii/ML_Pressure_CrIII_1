#Correlation Analysis
print('''\
------------------ ^^ ------------------
Step 2 for machine learning.
This program was written by Dawei Wen (温大尉)(*^^*).
If you have any question, please contact:
E-mail: ontaii@163.com.
Google Scholar: https://scholar.google.com/citations?hl=ja&user=U13L9sEAAAAJ
Research Gate: https://www.researchgate.net/profile/Dawei-Wen-2
------------------ ^_^ ------------------\
''')
import pandas as pd
Matrix = input('Enter the file name of the matrix with extension for machine learning, for example, Matrix.xlsx:\n')
Matrix = pd.read_excel(Matrix)
Matrix_Features = Matrix[['600to675 (%)', '675to700 (%)', '700to750 (%)', '750to950 (%)']]
print(Matrix_Features.corr())
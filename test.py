from numpy import *
import pandas as pd
dataset=pd.read_excel('Folds5x2_pp.xlsx')
X=dataset.iloc[:,:-1]
X=array(X)
print(X.shape)
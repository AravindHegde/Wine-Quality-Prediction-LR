import pandas as pd
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("wine_train.csv")
df2=pd.read_csv("wine_test.csv")
print(df1.head(10))
df1["content"]=df1["free sulfur dioxide"]/df1["total sulfur dioxide"]+df1["alcohol"]+df1["pH"]-df1["fixed acidity"]-df1["volatile acidity"]-df1["citric acid"]
ys = np.array(df1["quality"], dtype=np.float64)
xs = np.array(df1["content"], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)  
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
    
m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

r_squared = coefficient_of_determination(ys,regression_line)
print("r squared error",r_squared)

df2["content"]=df2["free sulfur dioxide"]/df2["total sulfur dioxide"]+df2["alcohol"]+df2["pH"]-df2["fixed acidity"]-df2["volatile acidity"]-df2["citric acid"]
predict_x = df2["content"]
df2["Predicted"] = ((m*predict_x)+b)
print("observation   quality")
print(df2["Predicted"])

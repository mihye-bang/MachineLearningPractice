from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")
# returns the first five rows
df.head()

x = df["height"]
y = df["weight"]
line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1, 1), y)

plt.plot(x, y, 'o')
plt.plot(x, line_fitter.predict(x.values.reshape(-1, 1)))
plt.show()


# fit : passing two variables for linear regression model
# reason for reshaping: x should be of 2 dimentional array. [[x1], [x2], [x3], ... , [xn]]


print(line_fitter.predict([[70]]))
print(line_fitter.coef_)
print(line_fitter.intercept_)

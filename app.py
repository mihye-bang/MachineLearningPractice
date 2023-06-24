from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pylplot as plt

line_fitter = LinearRegression()
# fit : passing two variables for linear regression model
line_fitter.fit(x, y)

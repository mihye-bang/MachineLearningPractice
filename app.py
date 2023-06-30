from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression():  # Linear Regression predicting weight based on height
    # read weight-height csv file
    df = pd.read_csv("heights.csv")
    # returns the first five rows
    df.head()
    # Features: input variables describing data
    x = df["height"]
    # Labels: variable we are predicting
    y = df["weight"]
    linear_fitter = LinearRegression()
    # .fit : passing two parameters/ variables (input, output) to train the linear regression model
    # When passing the x value, use '.values'. Otherwise it will also pass the headers too and give warnings
    # reason for reshape(-1,1): x should be in a 2-dimentional array like [[x1], [x2], [x3], ... [xn]] for potential Multiple Linear Regression
    linear_fitter.fit(x.values.reshape(-1, 1), y)

    # 'o' means a round marker (ro : red marker, b-- : blue dashes, s: square, ^ : triangle)
    plt.plot(x, y, 'o')
    # another plot to predict based on x input values. It will create a line
    plt.plot(x, linear_fitter.predict(x.values.reshape(-1, 1)))
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("SCATTER FOR THE LINEAR REGRESSION AND THE LINE OF PREDICTION")
    plt.show()

    # predicts output when x value 70 is provided
    print("Predicted output y(weight) when x(height) is 70: ",
          linear_fitter.predict([[70]]))
    # print coefficient (m)
    print("coefficient (m): ", linear_fitter.coef_)
    # print intercept (b)
    print("intercept (b): ", linear_fitter.intercept_)


def multiple_linear_regression():
    df = pd.read_csv("manhattan.csv")
    df.head()
    x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck',
            'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
    y = df[['rent']]

    # Segregate training set and test set using train_set_split method with ratio of 8:2
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2)

    # Create a model using training data
    mlr = LinearRegression()
    # When passing the x_train value, use '.values'. Otherwise it will also pass headers too and give warnings
    mlr.fit(x_train.values, y_train)

    # Predict the rent with example input
    my_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
    my_predict = mlr.predict(my_apartment)
    # Print and see the expected values
    print("Predicted rent value(y) with input example: ", my_predict)

    # Save the predicted y values with the x_test's values
    y_predict = mlr.predict(x_test.values)
    # Compare the actual y value with the prediction using Scatter. If they match, it will be a line. alpha : transparency for the dots (only for visual effects)
    plt.scatter(y_test, y_predict, alpha=0.4)
    plt.xlabel("Actual Rent")
    plt.ylabel("Predicted Rent")
    plt.title("MULTPILE LINEAR REGRESSION")
    plt.show()

    # Print coefficient (m)
    print("coefficient (m): ", mlr.coef_)
    # Print intercept (b)
    print("intercept (b): ", mlr.intercept_)

    # Investigate the relations (between rent and siqe_sqft, building_age_yrs ... one by one)
    plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
    plt.title("RELATION BETWEEN THE RENT AND THE SIZE")
    plt.xlabel("Size")
    plt.ylabel("Predicted Rent")
    plt.show()

    plt.scatter(df[['building_age_yrs']], df[['rent']], alpha=0.4)
    plt.title("RELATION BETWEEN THE RENT AND THE BUILDING AGE YEARS")
    plt.xlabel("Building age years")
    plt.ylabel("Predicted Rent")
    plt.show()

    # .score : to analyze the accuracy of the model. The closer to 1, the more accurate. 0.7 is normally considered to be good
    print(mlr.score(x_train, y_train))


def min_max_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst)-min(lst))
        normalized.append(normalized_num)
    # Pros: All scales of the features are the same
    # Cons: it gets affected too much by outlier
    return normalized


def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst) / np.std(lst))
        normalized.append(normalized_num)
    # Pros: it takes care of outlier
    # Cons: it does not produce the normalized data with exact same criteria
    return normalized


# linear_regression()
# multiple_linear_regression()

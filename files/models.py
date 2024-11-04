from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pandas as pd
from sklearn.linear_model import Ridge

def get_model(data):
    y = data['days'].dt.total_seconds() / 60
    X = data.drop(columns='days')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = Ridge().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = mean_absolute_error(y_pred, y_test)
    coef = model.coef_
    print("Acc:", acc)
    print("Coefs:", coef)

if __name__ == "__main__":
    interested = pd.read_csv("data.csv")
    get_model(interested)
# Linear Regression Health Costs Calculator - FCC ML with Python project 4
# Linear regression on insurance dataset; MAE target < 3500 (sklearn)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    # one-hot encode categoricals
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    y = df['expenses']
    X = df.drop(columns=['expenses'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return model, mae

if __name__ == '__main__':
    print('train_and_evaluate(csv_path) -> (model, MAE); expected MAE < 3500')

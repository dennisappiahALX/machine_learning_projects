import pandas as pd
from src import config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train(data):
    df = pd.read_csv(config.TRAIN_PROCESSED_DATA)
    MODELS = {
        'linearregression': LinearRegression(),
        'randomforest' : RandomForestClassifier(),
        'xgb': xgb.XGBRegressor()
    }

    X = df.iloc[:,:-1]
    y = df.SalePrice

    X_train, X_test, y_train,y_test = train_test_split(X,y, random_state= 10, test_size=0.3)

    model = MODELS[config.MODEL[2]]
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    # print(y_predict)

    print(config.MODEL[2], " accuracy: ", r2_score(y_test, y_predict))

train(config.TRAIN_PROCESSED_DATA)
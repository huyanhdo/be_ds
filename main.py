from typing import List
import math
from fastapi import FastAPI, Query
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

app = FastAPI()

linear = pickle.load(open('./model/linear.pickle','rb'))
lasso = pickle.load(open('./model/lasso.pickle','rb'))
linear = pickle.load(open('./model/ridge.pickle','rb'))

lgbm = pickle.load(open('./model/lgbm.pickle','rb'))
xgb = pickle.load(open('./model/xgb.pickle','rb'))

dt = pickle.load(open('./model/decision_tree.pickle','rb'))
rf = pickle.load(open('./model/random_forest.pickle','rb'))

def type_of_house(x):
    if x == 'Nhà biệt thự':
        return 3.0
    if x== 'Nhà phố liền kề':
        return 2.0
    if x == 'Nhà mặt phố':
        return 2.0
    if x == 'Nhà ngõ':
        return 1.0
    return np.nan

def preprocessing(features):
    area = math.log(float(features['Area']))
    number_of_bed_room = float(features['Number_of_bedroom'])
    type_of_house =  type_of_house(features['Type_of_house'])
    useable_area = math.log(float(features['Useable_area']))
    return np.asarray([area,number_of_bed_room,type_of_house,useable_area])

@app.get("/")
def root():
    return {"Hello": "Data Science"}

@app.get('/model')
def predict(feature):
    feature = preprocessing(feature)
    pred = dt.predict(feature)
    return math.e**pred

# @app.get("/linear_regression/")
# def linear_regression(features: List[str] = Query(None)):
#     # pass parameters to model
#     return {"features": features}


# @app.get("/random_forest/")
# def random_forest(features: List[str] = Query(None)):
#     # pass parameters to model
#     return {"Features": features}


# @app.get("/bootsting_algorithm/")
# def bootsting_algorithm(features: List[str] = Query(None)):
#     # pass parameters to model
#     return {"Features": features}


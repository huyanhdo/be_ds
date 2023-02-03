from typing import List
import math
from fastapi import FastAPI, Query
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

class Data(BaseModel):
    Area:str
    Number_of_bedroom:str
    Type_of_house:str
    Useable_area:str

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

linear = pickle.load(open('./model/linear.pickle','rb'))
lasso = pickle.load(open('./model/lasso.pickle','rb'))
ridge = pickle.load(open('./model/ridge.pickle','rb'))

lgbm = pickle.load(open('./model/lgbm.pickle','rb'))
xgb = pickle.load(open('./model/xgb.pickle','rb'))

dt = pickle.load(open('./model/decision_tree.pickle','rb'))
rf = pickle.load(open('./model/random_forest.pickle','rb'))

def handle_type_of_house(x):
    # x = str(x,'utf-8')
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
    features = json.loads(features)
    area = math.log(float(features['Area']))
    number_of_bed_room = float(features['Number_of_bedroom'])
    type_of_house =  handle_type_of_house(features['Type_of_house'])
    useable_area = math.log(float(features['Useable_area']))
    fea = np.asarray([[area,number_of_bed_room,type_of_house,useable_area]])
    return fea

@app.get("/")
def root():
    return {"Hello": "Data Science"}

@app.post('/decision_tree')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = dt.predict(feature)[0]
    res = math.e**pred
    return res

@app.post('/random_forest')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = rf.predict(feature)[0]
    res = math.e**pred
    return res

@app.post('/xgboost')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = xgb.predict(feature)[0]
    res = math.e**pred
    return res

@app.post('/lightgbm')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = lgbm.predict(feature)[0]
    res = math.e**pred
    return res

@app.post('/linear_regression')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = linear.predict(feature)[0]
    res = math.e**pred
    return res

@app.post('/lasso_regression')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = lasso.predict(feature)[0]
    res = math.e**pred
    return res
@app.post('/ridge_regression')
def predict(feature:Data):
    feature = preprocessing(feature.json())
    # print(feature)
    pred = ridge.predict(feature)[0]
    res = math.e**pred
    return res
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


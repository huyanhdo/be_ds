from typing import List

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/")
def root():
    return {"Hello": "Data Science"}


@app.get("/linear_regression/")
def linear_regression(features: List[str] = Query(None)):
    # pass parameters to model
    return {"features": features}


@app.get("/random_forest/")
def random_forest(features: List[str] = Query(None)):
    # pass parameters to model
    return {"Features": features}


@app.get("/bootsting_algorithm/")
def bootsting_algorithm(features: List[str] = Query(None)):
    # pass parameters to model
    return {"Features": features}


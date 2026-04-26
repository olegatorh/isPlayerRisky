import sys

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import RedirectResponse, Response
from uvicorn import run as app_run

from src.exception.exception import RiskyException
from src.utils.main_utils import load_object
from src.utils.ml_utils.estimator import RiskyModel

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health" , tags=["health"])
async def health():
    try:
        return Response("ok", status_code=200)
    except Exception as e:
        raise RiskyException(e, sys)


@app.post("/predict", description="predict is player risky" )
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print(df.head())

        final_model = load_object("final_model/final_model.pkl")
        predict = final_model.predict(df)

        return Response(f"{predict.tolist()}", status_code=200)

    except Exception as e:
        raise RiskyException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ner.config.configuration import Configuration
from ner.pipeline.prediction_pipeline import PredictionPipeline
from ner.pipeline.train_pipeline import TrainPipeline

app = FastAPI()


@app.post("/train")
def train(request: Request):
    try:
        pipeline = TrainPipeline(Configuration())
        pipeline.run_pipeline()
        return JSONResponse(content="Training Completed", status_code=200)
    except Exception as e:
        raise JSONResponse(content={"Error while training pipeline"}, status_code=500)


@app.post("/predict")
def predict(request: Request, data):
    try:
        pipeline = PredictionPipeline(config=Configuration())
        response = pipeline.run_pipeline(data=data)
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise JSONResponse(content={"Error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

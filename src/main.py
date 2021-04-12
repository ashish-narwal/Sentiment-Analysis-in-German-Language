import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import File, UploadFile
import json


class ModelParams(BaseModel):
    text: str


app = FastAPI()

from .models import SentimentModel

model = SentimentModel(model_name="oliverguhr/german-sentiment-bert")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{text}")
async def predict(text: str):
    pred = model.predict_sentiment(text)
    return pred


@app.post("/predict-post/")
async def post_predict(data: ModelParams):
    pred = model.predict_sentiment(data.text)
    return pred


@app.post("/upload-text-file/")
async def text_upload_file(upload_file: UploadFile = File(...)):

    text_binary = await upload_file.read()
    contents = text_binary.decode()
    text = str(contents)
    pred = model.predict_sentiment(text)
    return pred


@app.post("/upload-json-file/")
async def json_upload_file(upload_file: UploadFile = File(...)):

    json_data = json.load(upload_file.file)
    text = str(json_data["text"])
    pred = model.predict_sentiment(text)
    return pred


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

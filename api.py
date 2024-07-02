import pandas as pd
from fastapi import FastAPI
import dill
from pydantic import BaseModel

app = FastAPI()
with open('./model/target-model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    Result: bool

@app.get('/status')
def status():
    return 'Ok'


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    y = model['model'].predict(df)

    return {
        'Result': y[0]
    }


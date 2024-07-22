import json

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
import dill
from pydantic import BaseModel

app = FastAPI()
with open('./model/target-model.pkl', 'rb') as file:
    model = dill.load(file)

model = model['model']

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
    Result: int


@app.get('/status')
def status():
    return 'Ok'


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    y = model.predict(df)

    return {
        'Result': int(y[0])
    }


@app.get('/get_test_json')
def get_json_for_api_test():
    with open('./data/examples/example-1.json', 'r') as jsonFile:
        data = json.load(jsonFile)

    return data


@app.get('/get_feature_imp', response_model=list)
def get_top_features(n: int = 20):

    try:
        # Предполагаем, что модель упакована в Pipeline
        if hasattr(model.named_steps['classifier'], 'coef_'):
            feature_weights = model.named_steps['classifier'].coef_[0]
        elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_weights = model.named_steps['classifier'].feature_importances_
        else:
            raise HTTPException(status_code=400,
                                detail="Model does not have 'coef_' or 'feature_importances_' attributes")

        # Получаем имена признаков после обработки
        preprocessor = model.named_steps['preprocessor']
        transformer = preprocessor.named_steps['column_transformer']

        # Извлекаем имена признаков
        numerical_features = transformer.transformers_[0][2]
        categorical_features = transformer.transformers_[1][1].get_feature_names_out()

        feature_names = list(numerical_features) + list(categorical_features)

        if len(feature_names) != len(feature_weights):
            raise HTTPException(status_code=500, detail="Feature names and weights length mismatch")

        # Создаем DataFrame для удобства работы
        df = pd.DataFrame({
            'feature': feature_names,
            'weight': feature_weights
        })

        # Сортируем DataFrame по весу и берем топ N признаков
        top_features = df.nlargest(n, 'weight')

        return top_features.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

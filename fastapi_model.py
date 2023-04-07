from fastapi import FastAPI
import uvicorn
import pandas as pd
import pickle as pk

app = FastAPI()

global model
model = pk.load(open('model.pk', 'rb'))


@app.get('/')
def root(N: int, P: int, K: int, temperature: float, humidity: float, ph: float, rainfall: float) -> str:
    X_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    label = {'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3,
             'pigeonpeas': 4, 'mothbeans': 5, 'mungbean': 6,
             'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
             'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13,
             'muskmelon': 14, 'apple': 15, 'orange': 16, 'papaya': 17,
             'coconut': 18, 'cotton': 19, 'jute': 20, 'coffee': 21}
    return list(label.keys())[list(label.values()).index(model.predict(X_data))]


if __name__ == '__main__':
    uvicorn.run('fastapi_model:app', reload=True)

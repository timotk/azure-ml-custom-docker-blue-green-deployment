from fastapi import FastAPI
from pydantic import BaseModel
from sklearn import linear_model

# We train a very simple model, since this is just an example
model = linear_model.LinearRegression()
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
model.fit(X, y)


app = FastAPI(title="ML Endpoint")


class PredictionRequest(BaseModel):
    values: list[list[float]]


class PredictionResponse(BaseModel):
    predictions: list[float]


@app.get("/health")
@app.get("/ready")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    y_pred = model.predict(request.values)
    return PredictionResponse(predictions=y_pred)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, model, MODEL_VERSION
from fastapi import HTTPException



app = FastAPI()



@app.get('/')
def home():
    return {'message':'Insurance Premium Prediction API'}

@app.get('/health')
def health_check():
    return {
        'status':'ok',
        'version':MODEL_VERSION,
        'model_loaded':model is not None
    }

@app.post('/predict', response_model=PredictionResponse)
def predict_premium(data: UserInput):

    try:
        prediction = predict_output(data.dict())

        return {
            "predicted_category": prediction["predicted_category"],
            "confidence": prediction["confidence"],
            "class_probabilities": prediction["class_probabilities"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
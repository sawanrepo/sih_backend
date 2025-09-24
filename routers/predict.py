from fastapi import APIRouter
from models.predict_models import PredictRequest, PredictResponse
from core.imputer import predict_missing

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    data = predict_missing(req.dict())
    predicted_fields = [k for k, v in req.dict().items() if v is None]
    
    return PredictResponse(
        success=True,
        data=data,
        predicted_fields=predicted_fields,
        confidence_scores={field: 0.85 for field in predicted_fields}  # TODO: add real scores
    )
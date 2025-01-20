from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
from src.schemas.data_schemas import ModelData
from src.core.recommendation_engine import RecommendationEngine
from src.services.load_data import load_data

app = FastAPI()
engine = RecommendationEngine()

# Initialize the engine with data
df = load_data("data/model_inventory.csv")
engine.fit(df)

@app.post("/add_model/")
async def add_model(model_data: ModelData):
    try:
        engine.add_model(model_data.dict())
        return {"message": "Model added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommend/")
async def get_recommendations(query: str, top_n: int = 5):
    try:
        recommendations = engine.get_recommendations(query, top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/")
async def get_all_models():
    return {"models": engine.df.to_dict('records')}

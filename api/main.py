from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import RecommendRequest, RecommendResponse
from model_loader import get_recommendations
import time

app = FastAPI(title="CartMatch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        results = get_recommendations(req.product_name, req.catalog, req.top_n)
        return RecommendResponse(
            product_name=req.product_name,
            recommendations=results,
            total_catalog_size=len(req.catalog)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

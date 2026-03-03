'''Initial point for service mode'''

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Tender RAG Assistant")
app.include_router(router)
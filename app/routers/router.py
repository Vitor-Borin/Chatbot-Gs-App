from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.enchente_service import verificar_enchente_por_bairro
from app.langchain_utils import gerar_resposta

router = APIRouter()

class Pergunta(BaseModel):
    question: str

@router.post("/perguntar")
def responder(pergunta: Pergunta):
    resposta = gerar_resposta(pergunta.question)
    return resposta

@router.get("/enchente")
def consultar_enchente(bairro: str):
    try:
        alerta = verificar_enchente_por_bairro(bairro)
        return {"bairro": bairro, "alerta": alerta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

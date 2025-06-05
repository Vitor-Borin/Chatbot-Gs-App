from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.enchente_service import verificar_enchente_por_bairro
from app.langchain_utils import gerar_resposta

router = APIRouter()

class Pergunta(BaseModel):
    question: str
    menu_state: str = "main_menu"

@router.post("/perguntar")
def responder(pergunta: Pergunta):
    try:
        texto_resposta, audio_base64, novo_menu_state = gerar_resposta(pergunta.question, pergunta.menu_state)
        return {"resposta": texto_resposta, "audio": audio_base64, "menu_state": novo_menu_state}
    except Exception as e:
        print(f"Erro na rota /perguntar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/enchente")
def consultar_enchente(bairro: str):
    try:
        alerta = verificar_enchente_por_bairro(bairro)
        return {"bairro": bairro, "alerta": alerta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

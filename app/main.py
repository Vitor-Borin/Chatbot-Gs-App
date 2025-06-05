from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.langchain_utils import gerar_resposta

app = FastAPI()

# Permite acesso externo (necessário para o app React Native se comunicar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, defina domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define o corpo da requisição
class Pergunta(BaseModel):
    question: str

# Endpoint principal
@app.post("/perguntar")
def responder(pergunta: Pergunta):
    resposta, audio_base64 = gerar_resposta(pergunta.question)
    return {
        "resposta": resposta,
        "audio": audio_base64  # Base64 do áudio .mp3
    }

# Execução local com uvicorn (opcional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

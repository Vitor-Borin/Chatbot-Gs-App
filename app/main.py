from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import router

app = FastAPI()

app.include_router(router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
from app.database.connection import get_connection

@app.get("/teste-bd")
def testar_conexao():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM GS_BAIRRO FETCH FIRST 5 ROWS ONLY")
        bairros = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"status": "ok", "dados": bairros}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

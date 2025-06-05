from app.database.connection import get_connection

def verificar_enchente_por_bairro(bairro_usuario: str) -> str:
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT COUNT(*) 
        FROM GS_ALERTAS A
        JOIN GS_COORDENADAS C ON A.ID_COORDENADA = C.ID_COORDENADA
        JOIN GS_BAIRRO B ON C.ID_BAIRRO = B.ID_BAIRRO
        WHERE LOWER(B.NM_BAIRRO) = :bairro
    """
    cursor.execute(query, [bairro_usuario.lower()])
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    if count > 0:
        return f"⚠️ Alerta! Foram registrados riscos de enchente recentemente no bairro {bairro_usuario}. Fique atento e siga as recomendações de segurança."
    else:
        return f"✅ O bairro {bairro_usuario} não apresenta riscos de enchente no momento. Continue acompanhando os alertas."

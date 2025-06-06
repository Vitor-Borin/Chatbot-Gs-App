from app.database.connection import get_connection

def verificar_enchente_por_bairro_e_cidade(bairro_usuario: str, cidade_usuario: str) -> str:
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT A.DESC_ALERTA, N.DESC_NIVEL
        FROM GS_ALERTAS A
        JOIN GS_BAIRRO B ON A.FK_BAIRRO = B.ID_BAIRRO
        JOIN GS_CIDADE D ON B.FK_CIDADE = D.ID_CIDADE
        JOIN GS_NIVEL_ALERTA N ON A.FK_NIVEL = N.ID_NIVEL
        WHERE LOWER(B.NOM_BAIRRO) = LOWER(:bairro)
          AND LOWER(D.NOM_CIDADE) = LOWER(:cidade)
    """

    resposta = ""
    try:
        cursor.execute(query, [bairro_usuario, cidade_usuario])
        resultado = cursor.fetchone()

        if resultado:
            desc_alerta, desc_nivel = resultado
            resposta = f"⚠️ Alerta em {bairro_usuario}, {cidade_usuario}!\n\nNível do Alerta: {desc_nivel}\nDescrição: {desc_alerta}\n\nFique atento e siga as recomendações de segurança."
        else:
            resposta = f"✅ O bairro {bairro_usuario}, {cidade_usuario} não apresenta riscos de enchente no momento. Continue acompanhando os alertas."

    except Exception as e:
        print(f"Erro ao verificar enchente no BD para o bairro {bairro_usuario} e cidade {cidade_usuario}: {e}")
        resposta = f"Desculpe, ocorreu um erro ao verificar informações para o bairro {bairro_usuario}, {cidade_usuario}. Por favor, tente novamente mais tarde."
    finally:
        cursor.close()
        conn.close()

    return resposta

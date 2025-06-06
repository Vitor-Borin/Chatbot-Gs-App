import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, List
from pathlib import Path
import base64
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import VideoUnavailable
import random
from app.services.enchente_service import verificar_enchente_por_bairro, verificar_enchente_por_bairro_e_cidade

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = InMemoryVectorStore(embeddings)

urls = [
    "https://www.gov.br/defesacivil/pt-br/prevenir-e-preparar/tipos-de-desastres",
    "https://www.gov.br/defesacivil/pt-br/prevenir-e-preparar/planos-de-contingencia",
    "https://www.inmet.gov.br/portal/index.php?r=alerta/page&page=cartilhas",
    "https://www.inmet.gov.br/portal/index.php?r=home/page&page=perigos_meteorologicos",
    "https://www.ibge.gov.br/explica/desastres-naturais/enchentes.html",
    "https://www.ibge.gov.br/explica/desastres-naturais/deslizamentos.html",
    "https://www.gov.br/defesacivil/pt-br/prevenir-e-preparar/queimadas",
    "https://www.gov.br/defesacivil/pt-br/prevenir-e-preparar/terremotos",
    "https://www.gov.br/defesacivil/pt-br/prevenir-e-preparar/secas",
    "https://www.unicef.org/brazil/educacao-emergencias-criancas",
    "https://www.unicef.org/brazil/kit-de-emergencia-para-familias-em-desastres-naturais",
    "https://www.gov.br/cemaden/pt-br",
    "https://www.msf.org.br/o-que-fazemos/atuacao/desastres-naturais/",
    "https://www.gov.br/mdr/pt-br/ultimas-noticias/entenda-a-diferenca-entre-os-tipos-de-desastres-naturais-e-tecnologicos-registrados-no-brasil",
    "https://climainfo.org.br/2024/07/31/desastres-naturais-causam-perdas-globais-de-us-120-bilhoes-no-1o-semestre-de-2024/",
    "https://www.mcti.gov.br/acompanhe-o-mcti/noticias/2025/01/cemaden-registra-recorde-de-alertas-e-mais-de-1-6-mil-ocorrencias-de-desastre-no-brasil-em-2024",
    "https://www.cnseg.org.br/noticias/desastres-naturais-sao-um-grande-desafio-para-os-municipios-brasileiros"
]

youtube_links = [
    "https://youtu.be/aIR6khgsc3A",
    "https://youtu.be/mnhJmOdoFEI"
]

docs = []
loader = UnstructuredURLLoader(urls=urls)
docs += loader.load()

for link in youtube_links:
    try:
        yt_loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)
        transcript_list = YouTubeTranscriptApi.get_transcript(yt_loader.video_id, languages=["pt"])
        transcript = " ".join([entry["text"] for entry in transcript_list])
        docs.append(Document(page_content=transcript, metadata=yt_loader._metadata))
    except Exception as e:
        print(f"Erro ao carregar transcrição do YouTube {link}: {e}")
        pass

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(all_splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    menu_state: str

def retrieve(state: State):
    if state.get("menu_state") in ["main_menu", "submenu_1", "awaiting_bairro", "reporting_problem", "prevention_tips"]:
         return {"context": state["context"], "menu_state": state["menu_state"]}

    all_retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    num_docs_to_use = min(5, len(all_retrieved_docs))
    selected_docs = random.sample(all_retrieved_docs, num_docs_to_use)
    return {"context": selected_docs, "menu_state": state["menu_state"]}


prompt = hub.pull("rlm/rag-prompt")

def generate(state: State):
    pergunta = state["question"].strip().lower()
    current_state = state.get("menu_state", "initial")

    main_menu_answer = (
        "👋 Olá, somos o Drenna!\n\n"
        "Selecione a categoria que melhor define sua dúvida:\n\n"
        "1 - Alertas e Situações de Risco\n"
        "2 - Relatar um problema\n"
        "3 - Prevenção e Dicas"
    )

    submenu_1_answer = (
        "Entendido! Agora escolha uma das opções abaixo:\n\n"
        "1 - Verificar enchentes na minha região\n"
        "2 - Receber alertas e notificações\n"
        "3 - Voltar ao Menu Principal"
    )

    if current_state != "initial" and pergunta in ["voltar", "menu", "menu principal", "cancelar"]:
        return {
            "answer": main_menu_answer,
            "menu_state": "main_menu"
        }

    if current_state == "initial":
        return {
            "answer": main_menu_answer,
            "menu_state": "main_menu"
        }

    elif current_state == "main_menu":
        if pergunta == "1":
            return {
                "answer": submenu_1_answer,
                "menu_state": "submenu_1"
            }

        if pergunta == "2":
            return {
                "answer": "Entendido! Por favor, me diga qual problema você quer relatar.",
                "menu_state": "reporting_problem"
            }

        if pergunta == "3":
            return {
                "answer": (
                    "Como posso te ajudar hoje?\n\n"
                    "Você pode perguntar, por exemplo:\n"
                    "- Quais itens são essenciais em uma enchente?\n"
                    "- O que fazer antes de um deslizamento?\n"
                    "- Onde buscar ajuda na minha região?"
                ),
                "menu_state": "prevention_tips"
            }
        return {
            "answer": "Opção inválida. " + main_menu_answer,
            "menu_state": "main_menu"
        }

    elif current_state == "submenu_1":
        if pergunta == "1":
            return {"answer": "Beleza! Você pode me dizer o nome do seu bairro e cidade?", "menu_state": "awaiting_bairro"}

        if pergunta == "2":
            return {"answer": "✅ Pronto! Seu aplicativo foi configurado para enviar alertas e notificações automaticamente.", "menu_state": "main_menu"}

        if pergunta == "3":
             return {
                "answer": main_menu_answer,
                "menu_state": "main_menu"
            }
        
        return {
            "answer": "Opção inválida. " + submenu_1_answer,
            "menu_state": "submenu_1"
        }

    elif current_state == "awaiting_bairro":
        if pergunta in ["menu", "voltar", "menu principal", "cancelar"]:
             return {
                "answer": main_menu_answer,
                "menu_state": "main_menu"
            }

        # Tenta extrair bairro e cidade da pergunta do usuário
        partes = pergunta.split(',')
        if len(partes) == 2:
            bairro_recebido = partes[0].strip()
            cidade_recebida = partes[1].strip()
            try:
                # Chama a nova função que busca alerta e nível
                resposta_bd = verificar_enchente_por_bairro_e_cidade(bairro_recebido, cidade_recebida)
            except Exception as e:
                print(f"Erro ao chamar verificar_enchente_por_bairro_e_cidade: {e}")
                resposta_bd = "Desculpe, ocorreu um erro ao processar sua solicitação. Por favor, tente novamente mais tarde."
        else:
            resposta_bd = "Formato inválido. Por favor, digite o bairro e a cidade separados por vírgula (ex: Centro, São Paulo)."

        final_resposta_bd = resposta_bd + "\n\nDigite 'menu' ou 'voltar' para retornar ao Menu Principal."

        return {"answer": final_resposta_bd, "menu_state": "main_menu"} # Retorna ao menu principal após a resposta

    elif current_state == "reporting_problem":
        if pergunta in ["menu", "voltar", "menu principal", "cancelar"]:
             return {
                "answer": main_menu_answer,
                "menu_state": "main_menu"
            }
        problema_relatado = pergunta 
        print(f"Problema relatado: {problema_relatado}") 

        confirmacao_resposta = "Obrigado! Sua solicitação foi registrada e será analisada por nossa equipe.\n\nDigite 'menu' ou 'voltar' para retornar ao Menu Principal."

        return {"answer": confirmacao_resposta, "menu_state": "main_menu"}

    elif current_state == "prevention_tips":
        if pergunta in ["menu", "voltar", "menu principal", "cancelar"]:
             return {
                "answer": main_menu_answer,
                "menu_state": "main_menu"
            }

        docs_content = "\n\n".join(doc.page_content for doc in state.get("context", []))

        has_list_keywords = any(p in pergunta for p in ["o que levar", "kit", "itens", "essenciais", "emergência"])
        has_sim_keywords = any(p in pergunta for p in ["simule", "etapa", "passo a passo"])

        if has_list_keywords and has_sim_keywords:
            docs_content += (
                "\n\nResponda como uma simulação realista dividida em etapas claras:"
                "ANTES, DURANTE E DEPOIS do desastre. Na etapa ANTES, inclua uma lista clara de itens essenciais separada por hífens (-), sem explicações longas para a lista."
                "Seja direto, didático e empático."
            )
        elif has_sim_keywords:
            docs_content += (
                "\n\nResponda como uma simulação realista dividida em etapas claras:"
                "ANTES, DURANTE E DEPOIS do desastre. Seja direto, didático e empático."
            )
        elif has_list_keywords:
             docs_content += (
                "\n\nMonte uma resposta iniciando com a frase: "
                "'A lista de itens essenciais para essa emergência é a seguinte:', "
                "seguida por uma lista clara separada por hífens (-), sem explicações longas."
             )

        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })

        try:
            response = llm.invoke(messages)
            resposta_original = response.content.strip()
        except Exception as e:
             print(f"Erro ao invocar LLM com pergunta '{state['question']}': {e}")
             resposta_original = "Desculpe, não consegui processar sua pergunta no momento. Tente perguntar de outra forma ou digite 'menu' para retornar ao menu principal."

        resposta_final = resposta_original + (\
            "\n\nDigite 'menu' ou 'voltar' a qualquer momento para retornar ao Menu Principal."\
        )

        return {"answer": resposta_final, "menu_state": "prevention_tips"}

    print(f"Estado inesperado: {current_state}") 
    return {"answer": "Desculpe, algo inesperado aconteceu. Por favor, digite 'menu' para recomeçar.", "menu_state": "main_menu"}


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gerar_audio(texto: str) -> str:
    caminho = "resposta.mp3"
    if not texto.strip():
        return None 

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=texto,
        ) as resposta:
            resposta.stream_to_file(caminho)

        with open(caminho, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return audio_base64
    except Exception as e:
        print(f"Erro ao gerar áudio para o texto: {texto[:50]}... Erro: {e}") 
        return None 


def gerar_resposta(pergunta: str, incoming_menu_state: str = "initial"):
    entrada = {
        "question": pergunta,
        "context": [],
        "answer": "", 
        "menu_state": incoming_menu_state 
    }

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve) 
    graph_builder.add_node("generate", generate) 

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    graph = graph_builder.compile()

    resultado = graph.invoke(entrada)

    resposta_texto = resultado["answer"]
    novo_menu_state = resultado["menu_state"]

    audio_base64 = gerar_audio(resposta_texto)

    return resposta_texto, audio_base64, novo_menu_state
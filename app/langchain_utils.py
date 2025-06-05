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
    "https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/noticias/2025/01/cemaden-registra-recorde-de-alertas-e-mais-de-1-6-mil-ocorrencias-de-desastre-no-brasil-em-2024",
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
    all_retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    num_docs_to_use = min(5, len(all_retrieved_docs))
    selected_docs = random.sample(all_retrieved_docs, num_docs_to_use)
    return {"context": selected_docs, "menu_state": state["menu_state"]}

prompt = hub.pull("rlm/rag-prompt")

def generate(state: State):
    pergunta = state["question"].strip().lower()
    current_state = state.get("menu_state", "initial")

    print(f"Current State: {current_state}, Question: {pergunta}")

    # Check for main menu return regardless of current state (except initial)
    if current_state != "initial" and pergunta in ["voltar", "menu", "menu principal", "cancelar"]:
        return {
            "answer": (
                "👋 Olá, somos o Drenna!\n\n"
                "Selecione a categoria que melhor define sua dúvida:\n\n"
                "1 - Alertas e Situações de Risco\n"
                "2 - Relatar um problema\n"
                "3 - Prevenção e Dicas"
            ),
            "menu_state": "main_menu"
        }

    # Handle initial state
    if current_state == "initial":
        # Qualquer entrada na estado inicial leva ao menu principal
        return {
            "answer": (
                "👋 Olá, somos o Drenna!\n\n"
                "Selecione a categoria que melhor define sua dúvida:\n\n"
                "1 - Alertas e Situações de Risco\n"
                "2 - Relatar um problema\n"
                "3 - Prevenção e Dicas"
            ),
            "menu_state": "main_menu"
        }

    # Handle main menu state
    elif current_state == "main_menu":
        if pergunta == "1":
            return {
                "answer": (
                    "Entendido! Agora escolha uma das opções abaixo:\n\n"
                    "1 - Verificar enchentes na minha região\n"
                    "2 - Receber alertas e notificações"
                ),
                "menu_state": "submenu_1"
            }

        if pergunta == "2":
            return {"answer": "Entendido! Por favor, me diga qual problema você quer relatar.", "menu_state": "reporting_problem"}

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

        # If input is not a valid main menu option, show the main menu again
        return {
            "answer": (
                "Opção inválida. Por favor, escolha uma das opções disponíveis:\n\n"
                "1 - Alertas e Situações de Risco\n"
                "2 - Relatar um problema\n"
                "3 - Prevenção e Dicas"
            ),
            "menu_state": "main_menu"
        }

    # Handle submenu 1 state (Alertas e Situações de Risco)
    elif current_state == "submenu_1":
        if pergunta == "1":
            return {"answer": "Beleza! Você pode me dizer o nome do seu bairro e cidade?", "menu_state": "awaiting_bairro"}

        if pergunta == "2":
            return {"answer": "✅ Pronto! Seu aplicativo foi configurado para enviar alertas e notificações automaticamente.", "menu_state": "main_menu"}

        # Add a numbered option to return to the main menu
        if pergunta == "3":
             return {
                "answer": (
                    "👋 Olá, somos o Drenna!\n\n"
                    "Selecione a categoria que melhor define sua dúvida:\n\n"
                    "1 - Alertas e Situações de Risco\n"
                    "2 - Relatar um problema\n"
                    "3 - Prevenção e Dicas"
                ),
                "menu_state": "main_menu"
            }

        # If input is not a valid submenu 1 option, show the submenu again
        return {
            "answer": (
                "Opção inválida. Por favor, escolha uma das opções abaixo:\n\n"
                "1 - Verificar enchentes na minha região\n"
                "2 - Receber alertas e notificações\n"
                "3 - Voltar ao Menu Principal"
            ),
            "menu_state": "submenu_1"
        }

    # Handle awaiting bairro state
    elif current_state == "awaiting_bairro":
         # Add a numbered option to return to the main menu
        if pergunta == "4": # Using 4 as an example, adjust if needed based on context
             return {
                "answer": (
                    "👋 Olá, somos o Drenna!\n\n"
                    "Selecione a categoria que melhor define sua dúvida:\n\n"
                    "1 - Alertas e Situações de Risco\n"
                    "2 - Relatar um problema\n"
                    "3 - Prevenção e Dicas"
                ),
                "menu_state": "main_menu"
            }
        bairro_recebido = pergunta
        try:
            from app.services.enchente_service import verificar_enchente_por_bairro
            resposta_bd = verificar_enchente_por_bairro(bairro_recebido)
        except Exception as e:
            resposta_bd = f"Desculpe, ocorreu um erro ao verificar o bairro {bairro_recebido}. Por favor, tente novamente mais tarde."

        return {"answer": resposta_bd, "menu_state": "main_menu"}

    # Handle reporting problem state
    elif current_state == "reporting_problem":
         # Add a numbered option to return to the main menu
        if pergunta == "4": # Using 4 as an example, adjust if needed based on context
             return {
                "answer": (
                    "👋 Olá, somos o Drenna!\n\n"
                    "Selecione a categoria que melhor define sua dúvida:\n\n"
                    "1 - Alertas e Situações de Risco\n"
                    "2 - Relatar um problema\n"
                    "3 - Prevenção e Dicas"
                ),
                "menu_state": "main_menu"
            }
         # Assuming any input here is the problem description
        return {"answer": "Obrigado! Sua solicitação foi registrada e será analisada por nossa equipe.", "menu_state": "main_menu"}

    # Handle prevention tips state
    elif current_state == "prevention_tips":
        # Add a numbered option to return to the main menu
        if pergunta == "4": # Using 4 as an example, adjust if needed based on context
             return {
                "answer": (
                    "👋 Olá, somos o Drenna!\n\n"
                    "Selecione a categoria que melhor define sua dúvida:\n\n"
                    "1 - Alertas e Situações de Risco\n"
                    "2 - Relatar um problema\n"
                    "3 - Prevenção e Dicas"
                ),
                "menu_state": "main_menu"
            }
        # Process question using RAG
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        response = llm.invoke(messages)
        resposta_original = response.content.strip()

        # Add instruction to return to main menu
        resposta_final = resposta_original + (
            "\n\nDigite 'menu' ou 'voltar' a qualquer momento para retornar ao Menu Principal."
        )

        # Stay in prevention_tips state to allow follow-up questions
        return {"answer": resposta_final, "menu_state": "prevention_tips"}

    # Fallback for unexpected states
    return {"answer": "Desculpe, algo inesperado aconteceu. Por favor, digite 'menu' para recomeçar.", "menu_state": "main_menu"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gerar_audio(texto: str) -> str:
    caminho = "resposta.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=texto,
        instructions="Responda de forma clara e empática como um assistente de emergência real"
    ) as resposta:
        resposta.stream_to_file(caminho)

    with open(caminho, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    return audio_base64

def gerar_resposta(pergunta: str, incoming_menu_state: str = "initial"):
    entrada = {
        "question": pergunta,
        "context": [],
        "answer": "",
        "menu_state": incoming_menu_state
    }
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    resultado = graph.invoke(entrada)
    resposta = resultado["answer"]
    novo_menu_state = resultado["menu_state"]
    audio_base64 = gerar_audio(resposta)

    return resposta, audio_base64, novo_menu_state
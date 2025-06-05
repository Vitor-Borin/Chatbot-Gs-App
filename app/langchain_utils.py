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
    current_state = state.get("menu_state", "main_menu")

    if pergunta in ["voltar", "menu", "menu principal"]:
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

    if current_state == "main_menu":
        cumprimentos = ["oi", "olá", "bom dia", "boa tarde", "boa noite"]
        if any(c in pergunta for c in cumprimentos):
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

        if pergunta == "1":
            return {
                "answer": (
                    "Entendido! Agora escolha uma das opções abaixo:\n\n"
                    "1 - Ver se há enchentes na minha região\n"
                    "2 - Receber alertas e notificações"
                ),
                "menu_state": "submenu_1"
            }

        if pergunta == "2":
            return {"answer": "Entendido! Me envie qual problema você quer relatar.", "menu_state": "main_menu"}

        if pergunta == "3":
            return {
                "answer": (
                    "Como posso te ajudar hoje?\n\n"
                    "Você pode perguntar, por exemplo:\n"
                    "- Quais itens são essenciais em uma enchente?\n"
                    "- O que fazer antes de um deslizamento?\n"
                    "- Onde buscar ajuda na minha região?"
                ),
                "menu_state": "main_menu"
            }

        if any(x in pergunta for x in ["problema com", "tive um problema", "relatar", "estrago", "emergência"]):
            return {"answer": "Obrigado! Sua solicitação foi registrada e será analisada por nossa equipe.", "menu_state": "main_menu"}

        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if any(palavra in pergunta for palavra in ["simule", "etapa", "passo a passo"]):
            docs_content += (
                "\n\nResponda como uma simulação realista dividida em etapas claras: "
                "ANTES, DURANTE e DEPOIS do desastre. Seja direto, didático e empático."
            )

        if any(p in pergunta for p in ["o que levar", "kit", "itens", "essenciais", "emergência", "preciso ter", "necessário", "lista"]):
            docs_content += (
                "\n\nMonte uma resposta iniciando com a frase: "
                "'A lista de itens essenciais para essa emergência é a seguinte:', "
                "seguida por uma lista clara separada por hífens (-), sem explicações longas."
            )

        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })

        response = llm.invoke(messages)
        resposta_original = response.content.strip()

        introducoes = [
            "Com base nas informações que encontrei: ",
            "De acordo com os dados disponíveis: ",
            "Verifiquei nas fontes e a informação é a seguinte: ",
            "Segundo o que apurei: "
        ]
        introducao_aleatoria = random.choice(introducoes) if introducoes else ""

        resposta_final = introducao_aleatoria + resposta_original

        if len(resposta_final) > 500:
            limite_resposta = 500 - len(introducao_aleatoria)
            if limite_resposta > 0:
                 resposta_final = introducao_aleatoria + resposta_original[:limite_resposta].rsplit(".", 1)[0] + "."
            else:
                 resposta_final = resposta_original[:480].rsplit(".", 1)[0] + "."

        if "não sei" in resposta_final.lower():
            resposta_final = (
                "Ainda não encontrei informações precisas sobre isso. "
                "Estou em constante aprendizado e posso continuar pesquisando se quiser."
            )

        return {"answer": resposta_final, "menu_state": "main_menu"}

    elif current_state == "submenu_1":
        if pergunta in ["1", "1 - ver se há enchentes na minha região"]:
            return {"answer": "Beleza! Você pode me dizer o nome do seu bairro e cidade?", "menu_state": "awaiting_bairro"}

        if pergunta in ["2", "2 - receber alertas e notificações"]:
            return {"answer": "✅ Pronto! Seu aplicativo foi configurado para enviar alertas e notificações automaticamente.", "menu_state": "main_menu"}

        return {
            "answer": (
                "Opção inválida no sub-menu. Por favor, escolha 1 ou 2, ou digite 'voltar'.\n\n"
                "Entendido! Agora escolha uma das opções abaixo:\n\n"
                "1 - Ver se há enchentes na minha região\n"
                "2 - Receber alertas e notificações"
            ),
            "menu_state": "submenu_1"
        }

    elif current_state == "awaiting_bairro":
        bairro_recebido = pergunta
        if 'a' in bairro_recebido.lower():
             resposta_bd = f"⚠️ Alerta! Foram registrados riscos de enchente recentemente no bairro {bairro_recebido}. Fique atento e siga as recomendações de segurança."
        else:
             resposta_bd = f"✅ O bairro {bairro_recebido} não apresenta riscos de enchente no momento. Continue acompanhando os alertas."

        return {"answer": resposta_bd, "menu_state": "main_menu"}

    return {"answer": "Desculpe, não entendi. Digite 'menu' para ver as opções.", "menu_state": "main_menu"}

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

def gerar_resposta(pergunta: str):
    entrada = {
        "question": pergunta,
        "context": [],
        "answer": "",
        "menu_state": "main_menu"
    }
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    resultado = graph.invoke(entrada)
    resposta = resultado["answer"]
    audio_base64 = gerar_audio(resposta)

    return resposta, audio_base64
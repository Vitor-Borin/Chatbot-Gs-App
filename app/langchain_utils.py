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

# Inicializa LLM e embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = InMemoryVectorStore(embeddings)

# Fontes web confiáveis sobre desastres naturais
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

# Vídeos com transcrição automática em português
youtube_links = [
    "https://youtu.be/aIR6khgsc3A",
    "https://youtu.be/mnhJmOdoFEI"
]

# Carregamento inicial
docs = []

loader = UnstructuredURLLoader(urls=urls)
docs += loader.load()

for link in youtube_links:
    try:
        yt_loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)
        transcript_list = YouTubeTranscriptApi.get_transcript(yt_loader.video_id, languages=["pt"])
        transcript = " ".join([entry["text"] for entry in transcript_list])
        docs.append(Document(page_content=transcript, metadata=yt_loader._metadata))
        print(f"✅ Transcrição carregada: {link}")
    except Exception as e:
        print(f"❌ Erro ao processar {link}\nMotivo: {e}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store.add_documents(all_splits)

print(f"✅ {len(all_splits)} fragmentos foram indexados com sucesso!")

# Estrutura do estado
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Função de busca dos documentos mais parecidos
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Função que gera a resposta baseada no contexto
prompt = hub.pull("rlm/rag-prompt")

def generate(state: State):
    cumprimentos = ["oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí"]
    if any(c in state["question"].lower() for c in cumprimentos):
        return {
            "answer": (
                "Olá! 👋 Como posso te ajudar hoje?\n\n"
                "Você pode perguntar, por exemplo:\n"
                "- Quais itens são essenciais em uma enchente?\n"
                "- O que fazer antes de um deslizamento?\n"
                "- Onde buscar ajuda na minha região?"
            )
        }

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    if any(palavra in state["question"].lower() for palavra in ["simule", "etapa", "passo a passo"]):
        docs_content += (
            "\n\nResponda como uma simulação realista dividida em etapas claras: "
            "ANTES, DURANTE e DEPOIS do desastre. Seja direto, didático e empático."
        )

    if any(p in state["question"].lower() for p in ["o que levar", "kit", "itens", "essenciais", "emergência", "preciso ter", "necessário", "lista"]):
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

    if "não sei" in response.content.lower():
        resposta_final = (
            "Ainda não encontrei informações precisas sobre isso. "
            "Estou em constante aprendizado e posso continuar pesquisando se quiser."
        )
    else:
        resposta_final = response.content.strip()
        if len(resposta_final) > 500:
            resposta_final = resposta_final[:480].rsplit(".", 1)[0] + "."

    return {"answer": resposta_final}

# Cliente da OpenAI para gerar áudio
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

# Função principal para usar no FastAPI
def gerar_resposta(pergunta: str):
    entrada = {
        "question": pergunta,
        "context": [],
        "answer": ""
    }
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    resultado = graph.invoke(entrada)
    resposta = resultado["answer"]
    audio_base64 = gerar_audio(resposta)

    return resposta, audio_base64
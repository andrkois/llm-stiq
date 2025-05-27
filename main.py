import gradio as gr
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_community.document_loaders.csv_loader import CSVLoader

# configuration
DATA_PATH = "data/ecommerce"
CHROMA_PATH = "chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document
from pathlib import Path

pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
pdf_docs = pdf_loader.load()

# Load all CSVs in the directory
csv_docs = []
for csv_path in Path(DATA_PATH).rglob("*.csv"):
    loader = CSVLoader(file_path=str(csv_path))
    csv_docs.extend(loader.load())

# Combine them
raw_documents = pdf_docs + csv_docs

# Add filename to metadata
for doc in raw_documents:
    if "source" in doc.metadata:
        doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]


# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)

import os
import csv
import time
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request
from starlette.config import Config
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
import gradio as gr

# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "a_very_secret_key"))

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
config_data = {'GOOGLE_CLIENT_ID': GOOGLE_CLIENT_ID, 'GOOGLE_CLIENT_SECRET': GOOGLE_CLIENT_SECRET}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# User folder mapping
USER_DATA_ACCESS = {
    "a.kois.anastasiadis@stiq.gr": "data/ecommerce",
    "a.lasdinou@stiq.gr": "data/hr",
}

# Logging setup
LOG_FILE = "/content/drive/MyDrive/Internal LLM/login_logout_log.csv"
log_path = Path(LOG_FILE)
if not log_path.exists():
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "action", "email"])

def log_event(email: str, action: str):
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.utcnow().isoformat(), action, email])

# User authentication and routing
def get_user(request: Request):
    return request.session.get('user')

def is_valid_email(email: str) -> bool:
    return email.endswith('@stiq.gr')

@app.get('/')
def public(user: dict = Depends(get_user)):
    return RedirectResponse(url='/gradio' if user else '/login-demo')

@app.route('/logout')
async def logout(request: Request):
    user = request.session.get('user')
    if user:
        log_event(user.get("email"), "logout")
    request.session.pop('user', None)
    response = RedirectResponse(url="/")
    response.delete_cookie("session")  # Example: delete session cookie
    return response

@app.route('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.route('/auth')
async def auth(request: Request):
    try:
        access_token = await oauth.google.authorize_access_token(request)
        user_info = dict(access_token)['userinfo']
    except OAuthError:
        return RedirectResponse(url='/')

    email = user_info.get('email')
    if not is_valid_email(email):
        return RedirectResponse(url='/error')

    data_path = USER_DATA_ACCESS.get(email)
    if not data_path:
        return RedirectResponse(url='/unauthorized')

    user_info['data_path'] = data_path
    request.session['user'] = user_info
    log_event(email, "login")
    return RedirectResponse(url='/')

@app.route('/error')
def error():
    return "You must use a valid @stiq.gr email to log in."

@app.route('/unauthorized')
def unauthorized():
    return "You are not authorized to access any resources."

# Login page
with gr.Blocks() as login_demo:
    gr.HTML("""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        flex-direction: column;
        text-align: center;
        padding: 16px;
    ">
        <h1 style="font-size: clamp(24px, 5vw, 36px);">👋 Welcome to Stiqchat</h1>
        <p style="font-size: clamp(16px, 3vw, 20px); color: gray;">Sign in with your Stiq email to continue</p>
        <div style="margin-top: 30px;">
            <a href="/login">
                <button style="
                    width: 100%;
                    max-width: 250px;
                    padding: 12px 20px;
                    font-size: 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                ">
                    🔐 Login with Google
                </button>
            </a>
        </div>
    </div>
    """)





# Mount the login page
app = gr.mount_gradio_app(app, login_demo, path="/login-demo")


# Vector DB setup
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
CHROMA_PATH = "chroma_db"
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# Chat functions
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

def get_documents_for_user(folder):
    if not folder or not os.path.exists(folder):
        return []
    loader = PyPDFDirectoryLoader(folder)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_docs)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    return chunks

def greet(request: gr.Request):

    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")

    first_name = user.get("given_name", "there")
    return f"""
    <div style='text-align: center; font-size: 24px;'>
        Καλώς ήρθες στο Stiqchat, <strong>{first_name}</strong>!
    </div>
    """

def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"]:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown file")
        knowledge += f"[Source: {filename}]\n{doc.page_content}\n\n"



    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant that answers questions based on knowledge provided in the "The knowledge" section.
        For each piece of information, mention the source filename as shown in the knowledge (e.g., [Source: returns_policy.pdf]).
        Do not use any internal knowledge — only what's in "The knowledge".

        The question: {message}

        Conversation history: {history}

        The knowledge:
        {knowledge}
        """


        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

def bot(history: list):
    user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if not user_msg:
        return history
    history.append({"role": "assistant", "content": " "})
    for chunk in stream_response(user_msg, history):
        history[-1]["content"] = chunk
        time.sleep(0.05)
        yield history

# Gradio interface
import random

greetings = [
    "Πως μπορώ να σε βοηθήσω;",
    "Καλησπέρα! Τι μπορώ να κάνω για σένα;",
    "Χρειάζεσαι βοήθεια με κάτι συγκεκριμένο;",
    "Πες μου τι σε απασχολεί!",
    "Είμαι εδώ για να σε βοηθήσω – τι θα ήθελες να μάθεις;"
]

initial_history = [
    {"role": "assistant", "content": random.choice(greetings)}
]

gr.HTML("""
<style>
    #logout-btn {
        background-color: #f44336 !important;
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        width: 50vw !important;
        margin: 20px auto;
        display: block;
        text-align: center;
    }

    #logout-btn:hover {
        background-color: #d32f2f;
    }

    #container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 16px;
    }
</style>
""")

with gr.Blocks(theme=gr.themes.Monochrome()) as main_demo:
    m = gr.Markdown()
    main_demo.load(greet, None, m)

    chatbot = gr.Chatbot(
        value=initial_history,
        type="messages",
        avatar_images=["user_avatar.png", "system_avatar.png"],
        show_copy_button=True,
        height=400,
        bubble_full_width=False
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["upload"],
    )

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    # Use block-level button to ensure it's centered
    logout_btn = gr.Button("🔓 Logout", link="/logout", elem_id="logout-btn")

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)

port = 8000
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

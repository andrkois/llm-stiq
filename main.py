import gradio as gr
from fastapi import FastAPI
from gradio.routes import App

# Minimal chatbot logic
def chatbot(message):
    return "You said: " + message

# Gradio Interface
demo = gr.ChatInterface(fn=chatbot)

# FastAPI wrapper
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

# Mount Gradio app under /chat
app.mount("/chat", App.create_app(demo))

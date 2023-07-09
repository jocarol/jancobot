import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from prompt import janco_prompt
from langchain import FAISS, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

def create_db_from_youtube_video_url(video_url):
    """
    Create a FAISS database from a YouTube video transcript.
    """
    loader = YoutubeLoader.from_youtube_url(video_url, language="fr")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(documents, embeddings)

    return db

def get_response_from_query(db, query):
    """
    Get a response from the LLMChain model for a given query.
    """
    documents = db.similarity_search(query, k=4)
    content = " ".join([d.page_content for d in documents])

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        janco_prompt)
    user_template = "{question}"
    user_message_prompt = HumanMessagePromptTemplate.from_template(
        user_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, user_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)

    response = chain.run(question=query, documents=content)
    response = response.replace("\n", "")

    return response

def init_db():
    """
    Initialize the FAISS database.
    """
    with st.spinner(text="Jancobot à bien reçu votre demande de consultation. Il va vous répondre dans quelques instants..."):
        db = create_db_from_youtube_video_url(video_url)
        st.success('JancoBot est en ligne !')
        return db

def app():
    """
    The main Streamlit application.
    """
    header = st.container()
    header.title("JancoBot 0.2")

    if 'db' not in st.session_state:
        st.session_state['db'] = init_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages:
        time.sleep(random.uniform(0.7, 1.5))
        with st.chat_message("jancoBot"):
            message_placeholder = st.image(
                'https://www.epargne-retraite-entreprises.bnpparibas.com/epargnants/Style%20Library/Styles/images/giphy.gif', width=80)
            greet = get_response_from_query(st.session_state.db, "/greet")
            message_placeholder.markdown(greet)
            st.session_state.messages.append(
                {"role": "jancoBot", "content": greet})

    if prompt := st.chat_input("Votre message"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        time.sleep(random.uniform(2, 4))
        with st.chat_message("jancoBot"):
            message_placeholder = st.image(
                'https://www.epargne-retraite-entreprises.bnpparibas.com/epargnants/Style%20Library/Styles/images/giphy.gif', width=80)
            jancoBot_response = get_response_from_query(
                st.session_state.db, prompt)
            message_placeholder.markdown(jancoBot_response)

        st.session_state.messages.append(
            {"role": "jancoBot", "content": jancoBot_response})

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    video_url = "https://www.youtube.com/watch?v=851Q-nPNx7I"
    embeddings = OpenAIEmbeddings()

    app()

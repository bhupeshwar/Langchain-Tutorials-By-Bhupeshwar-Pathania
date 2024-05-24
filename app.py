"""  
@Author : Bhupeshwar Pathania
@date : 23-05-2024
@Description : Q&A ChatBot Using OpenAI / Ollama(model="llama2") / Hugging face with Langchain

Run like this:
> PYTHONPATH=. streamlit run app.py   

"""
from langchain_openai import OpenAI
from langchain_community.llms import Ollama

from dotenv import load_dotenv
from langchain import HuggingFaceHub


load_dotenv() # take enviroment variables from .env

import streamlit as st
import os

## Function to laod OpenAI model and get response

def get_openia_response(question):
    llm=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),model_name="gpt-3.5-turbo-instruct", temperature=0.6)
    response=llm.invoke(question)
    return response

## Function to laod llama2 model and get response

def get_ollama_response(question):
    llm_ollama=Ollama(model="llama2")
    response=llm_ollama.invoke(question)
    return response

## Function to laod huggingface and get response

def get_huggingface_response(question):
    #llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0,"max_length":64})
    llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large")
    response=llm_huggingface.invoke(question)
    return response

## Initialize our streamit app

st.set_page_config(page_title="Q&A Demo")

st.header("Lanchain Application")

def get_text():
    input_text = st.text_input("Input: ", key="input")
    return input_text

user_input = get_text()

#input=st.text_input("Input: ", key="input")

"""
You can use get_huggingface_response / get_openia_response / get_ollama_response
as per your requirement
"""

if user_input:
    response=get_huggingface_response(user_input) 
    #response=get_openia_response(user_input) 
    #response=get_ollama_response(user_input)

submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    st.subheader("The Response is")
    st.write(response)

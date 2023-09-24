############################################################################################
############################################################################################
# App de interactuar con un CSV
############################################################################################
############################################################################################

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://betterprogramming.pub/build-a-chatbot-on-your-csv-data-with-langchain-and-openai-ed121f85f0cd
# https://www.youtube.com/watch?v=NzLqRYVYFME&ab_channel=ElTallerDeTD
# https://discuss.streamlit.io/t/speech-to-text-on-client-side-using-html5-and-streamlit-bokeh-events/7888


# https://www.youtube.com/watch?v=Z41pEtTAgfs&t=649s&ab_channel=Streamlit
# https://github.com/dataprofessor/openai-chatbot/blob/master/streamlit_app.py


# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/
# https://stackoverflow.com/questions/76264205/in-langchain-why-conversationalretrievalchain-not-remembering-the-chat-history
# https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db


# MEJORAS:
# Mostrar en el sidebar: boton de hablar para ingresar prompt



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

import pandas as pd
import numpy as np

import openai

from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain

import tempfile

from gtts import gTTS
import time

import streamlit as st

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def cadena_modelo(
  st_Archivo, # un archivo subido por streamlit
  st_ApiKey # una api key de openAI ingresada
):

  # rescatar ubicacion del archivo para cargarlo
  with tempfile.NamedTemporaryFile(delete=False) as archivo_temporal:
    archivo_temporal.write(st_Archivo.getvalue())
    ruta_archivo_temporal = archivo_temporal.name

  documento = PyPDFLoader(
    file_path=ruta_archivo_temporal
    ).load()
  
  # se crea objeto que hara division (https://github.com/langchain-ai/langchain/issues/1349)
  divisor_texto = RecursiveCharacterTextSplitter( # CharacterTextSplitter
    chunk_size=1000, # tamaño de cada bloque de extraccion
    chunk_overlap=50, # traslape de cada bloque
    separators=[' ',',','\n'] # separadores para eventualmente forzar separacion 
  )

  # se crean los objetos dividicos
  documentos = divisor_texto.split_documents(documento)
  
  # crear objeto de embedding
  mi_embedding = OpenAIEmbeddings(
    openai_api_key = st_ApiKey
  )

  # pasamos a base vectorial 
  base_vectorial = FAISS.from_documents(
    documents = documentos,
    embedding = mi_embedding
  )

  # creamos objeto de llm
  llm1 = ChatOpenAI(
    openai_api_key = st_ApiKey,
    temperature=0.0,
    model_name='gpt-3.5-turbo'
    )

  # creamos objeto cadena 
  cadena = ConversationalRetrievalChain.from_llm(
    llm = llm1, # modelo de lenguaje
    retriever = base_vectorial.as_retriever(), # base vectorial creada anteriormente
    return_source_documents = True # pedir que retorne documentos desde donde obtiene respuesta
    )

  # retornar entregables 
  return cadena


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('## :page_with_curl: Interactuar con un archivo PDF :page_with_curl:')

# autoria 
st.sidebar.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')

# ingresar OpenAI api key
usuario_api_key = st.sidebar.text_input(
  label='Tu OpenAI API key :key:',
  placeholder='Pega aca tu openAI API key',
  type='password'
  )

# subir archivo 
Archivo = st.sidebar.file_uploader('Subir Archivo pdf',type=['pdf'])


# colocar separador para mostrar en sidebar otras cosas
st.sidebar.markdown('---')


#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if Archivo:  
   
  #_____________________________________________________________________________
  # Aplicar todo lo relacionado al modelo de LLM para tener listas las consultas

  # ingresar api key del usuario rescatada anteriormente
  openai.api_key= usuario_api_key

  # creamos objeto cadena (usando funcion en cache creada antes)
  cadena_preguntas_pdf = cadena_modelo(
    st_Archivo = Archivo,
    st_ApiKey = usuario_api_key
    )

  # definimos lista de historial chat
  historial_chat = []
   
  
  if 'messages' not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

  if consulta := st.chat_input('Ingresa tu consulta aqui'):
    st.session_state.messages.append({'role': 'user', 'content': consulta})
    
    with st.chat_message('user'):
      st.markdown(consulta)
      
    with st.chat_message('assistant'):
      
      # calcular respuesta de llm
      entregable_llm = cadena_preguntas_pdf({
        'question': consulta,
        'chat_history': historial_chat # validar luego mejor opcion de guardar historial por costo de APi en buffer (memory=ConversationBufferWindowMemory(k=1)  https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
        })
      respuesta = entregable_llm['answer']
      
      # guardar respuesta en historial
      historial_chat.append([(consulta, respuesta)])
      
      # mostrar parte por parte (solo estetico, dado que puede mostrar mas rapidamente la respuesta de inmediato)
      mensaje_entregable = st.empty()
      respuesta_total = ''

      for r in respuesta.split(' '):        
        respuesta_total += r+' '
        mensaje_entregable.markdown(respuesta_total + '▌')
        time.sleep(0.07)

      mensaje_entregable.markdown(respuesta_total)
    
    # mostrar audio de respuesta
    audio_resultado = gTTS(
      text = respuesta,
      lang = 'es-us',
      slow = False
      )
    audio_resultado.save('audio_app_resultado.mp3')
    audio_resultado2 = open('audio_app_resultado.mp3', 'rb').read()
    st.sidebar.audio(audio_resultado2, format='mp3')
    
    # mostrar referencias desde donde se obtiene la respuesta
    with st.sidebar.expander('Referencias',expanded=False):
      for c in entregable_llm['source_documents']:
        st.markdown('##### Pagina: '+str(c.metadata['page']+1))
        st.write(c.page_content)
        st.write(' ')
        st.write(' ')
    
    
    st.session_state.messages.append({'role': 'assistant', 'content': respuesta_total})
    
    


# !streamlit run App_LLM_PDF3.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/23_Streamlit App chat PDF (11-09-23)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit

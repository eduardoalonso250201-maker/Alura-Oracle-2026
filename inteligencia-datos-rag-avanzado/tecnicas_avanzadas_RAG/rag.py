#Todo lo necesario para cargar las variables de ambiente 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser



load_dotenv() #se cargann las variables de entorno 

#se asignan a variables esas variables de entorno 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="langchain"

# Carga TODOS los archivos PDF que esten dentro de la carpeta 'documentos'
# glob='*.pdf' es el filtro: solo toma archivos con extension .pdf, ignora otros tipos
pdfs = DirectoryLoader('documentos', glob='*.pdf', loader_cls=PyPDFLoader).load()

# Carga el tokenizador del modelo de embeddings BAAI/bge-m3 (multilingue, popular para RAG)
# Este tokenizador NO genera embeddings, solo sirve para medir el texto en tokens
# (la misma unidad que usara el modelo de embeddings mas adelante)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

# Crea el divisor de texto (chunker) usando el tokenizador de HuggingFace como referencia,
# en lugar de medir el tamaño de cada chunk por caracteres simples
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,      # tokenizador que se usa para contar tokens por chunk
    chunk_size=1250,          # cada chunk tendra aproximadamente 1250 tokens
    chunk_overlap=150         # cada chunk comparte 150 tokens con el chunk vecino,
                               # para no perder contexto en los cortes entre fragmentos
)

#se hace el chuking con las condiciones definidas en "splitter"
fragmentos = splitter.split_documents(pdfs)

#Se crea la variable con el modelo de embedding de ollama, el cual es el mismo de hugging face
embeddings = OllamaEmbeddings(model='bge-m3:567m')

#AHORA SE CREARA LA VECTOR STORE que almacena los chunks y embeddings 
vector_store = FAISS.from_documents(documents=fragmentos, embedding=embeddings) 


#Ahora se configura toda la LLM
prompt = ChatPromptTemplate(
    [("system", """Responde usando exclusivamente el contenido que se incluye a continuacion.
Si la respuesta no esta en el contenido, indica que no tienes esa informacion.

#CONTENIDO
{contexto}"""),
     ("human", "{query}")]
)

retriever = vector_store.as_retriever() #esta linea toma el vector mas cercano a nuestra query de nuestra base de embedding
modelo = OllamaLLM(model = "gemma3:4b") #se indica el modelo de LLM
cadena = prompt | modelo | StrOutputParser() #se hace un langchain para cargar el promt al modelo LLM y que regrese solo el str de respuesta

pregunta = "Como solicitar el seguro de viaje?"


#REALIZAMOS EL MODELO DE INVOKE QUE USARA NUESTRA Y SU RESPECTIVA PREGUNTA
trechos = retriever.invoke(pregunta) #se recuperan los fragmentos mas similares a la pregunta, desde la base de vectores
contexto = "\n\n".join(trecho.page_content for trecho in trechos) #se unen los fragmentos en un solo texto, separados por doble salto de linea (generator expression, no lista ni diccionario)
#cadena.invoke({"query": pregunta, "contexto":contexto})#se aplica la cadena para que entre en accion la LLM

#almacenar respuesta en nuestra RAG
#se crea un diccionario con contexto y query
#el objetico de las siguientes lineas es que se haga una cadena para que todas las salidas se tomen como entradas del siguiente paso en lagnsmith
"""
rag_chain = (
     {"contexto": RunnablePassthrough() | retriever,#modulo que se tiene de langchain_core.runnable basicamente toma la entrada y la coloca como siguiente entrada para el siguiente paso de lacadena
     "query": RunnablePassthrough()}
    | prompt | modelo | StrOutputParser()
)

rag_chain.invoke(pregunta)
"""
#instanciar un modelo dencillo de ollama 
query_model = OllamaLLM(model="gemma3:4b")

#template para reescribir las query del usuario 

rewriter_prompt_template = """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta más precisa por medio de la búsqueda semántica.
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""

#se ejecuta la funcion que llama a la langhain pasando como contexto nuestro
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)

#se crea la cadena pero ahora para reescribir con llm la query del usuario y que la respuesta final solo sea el texto respuesta 
rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

#se invoca todo el proceso
#rewriter_chain.invoke(pregunta)

"""
rag_chain = (
    {
        "contexto": RunnablePassthrough() | rewriter_chain | retriever,
        "query":RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)

rag_chain.invoke(pregunta)
"""
#AHORA COMO HACER VARIAS PREGUNTAS CONSECUTIVAS yY RECIBIR UNA RESPUESTA CON BASE EN NUESTRA BASE DE VECTORES COMO CHATBOT 

#se puede hacer una lista de varias preguntas y usarla como iteraciones de inputs para nuestra cadena
#Con las siguientes lineas se podria omitir todo lo de rewriter_prompt_template, su cadena, su promt, etc, pero con solo comentar su linea de invoacion ya no hace nada 
template_multipregunta = """
Eres un asistente de modelo de lenguajes de IA. Tu tarea es generar cinco versiones diferentes de la pregunta 
del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar multiples
perspectivas sobre la pregunta del usuario, tu objetivo es auxiliar al usuario a superar algunas de las
limitaciones de la búsqueda por similitud basada en distancia. Debes generar únicamente las preguntas alternativas
separadas en filas diferentes (new line) sin ningún texto adicional.

# PREGUNTA ORIGINAL: {question}

# FORMATO DE SALIDA :
["primera pregunta","segunda pregunta",...,"quinta pregunta"]
"""
prompt_multipregunta = PromptTemplate.from_template(template_multipregunta)
chain_multipregunta = prompt_multipregunta | modelo | CommaSeparatedListOutputParser()

preguntas = chain_multipregunta.invoke({"question": pregunta})  
print(preguntas)

#Una vez que se generaron las cinco preguntas con base en la pregunta del usuario
#se invoca el rag-chain para que se corra cada una de las preguntass 

rag_chain = (
    {
        "contexto": RunnablePassthrough() | retriever,
        "query":RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)


for p in preguntas:
    #la respuesta a cada pregunta se ve como corrida separada en langsmith por caad pregunta 
    rag_chain.invoke(p)
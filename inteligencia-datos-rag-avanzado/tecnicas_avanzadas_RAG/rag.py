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
cadena.invoke({"query": pregunta, "contexto":contexto})#se aplica la cadena para que entre en accion la LLM
respuesta = cadena.invoke({"query": pregunta, "contexto": contexto}) #se ejecuta la cadena completa: prompt + LLM + parser
print(respuesta) 
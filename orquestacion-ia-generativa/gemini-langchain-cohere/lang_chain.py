# Integration with Google's Gemini chat models for LLM inference
from langchain_google_genai import ChatGoogleGenerativeAI
# Integration with Cohere's chat models for LLM inference
from langchain_cohere import ChatCohere
# Core message types used to build chat prompts
from langchain_core.messages import HumanMessage
# Model identifier for the Gemini Flash model
from my_models import GEMINI_FLASH
# API key for accessing the Gemini and cohere service
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
# Utility function that converts an image file into a base64 string
from my_helper import encode_image
#we can use the ChatPromptTemplate to create a prompt for the model, this is useful when we want to create a prompt with multiple messages or with variables
from langchain.prompts import ChatPromptTemplate

# Initialize the Cohere model with the API key
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY
)

# Initialize the Google Generative AI model with the API key and model name
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

#here we invoke the gemini model with a simple question, the model will return a response object that contains the content of the answer
respuesta = llm.invoke("Cuales canales colombianos de youtube me recomiendas para saber mas sobre telefonos inteligentes?")
print(f"Gemini: ", respuesta.content)



#here we invoke the cohere model with a simple question, the model will return a response object that contains the content of the answer
#On the documentation of each API you can find how to call the model, response format, and other parameters 
respuesta = llm.invoke([HumanMessage(content="Cuales canales colombianos de youtube me recomiendas para saber mas sobre telefonos inteligentes?")])
print(f"Cohere: ", respuesta.content)


#we can encode the image using the helper function
imagen = encode_image('datos/ejemplo_grafico.jpg')

#prompt
pregunta = "describe la imagen: "

# Create a HumanMessage with the prompt and the encoded image
#this is the format that the model expects for multimodal inputs
#as we did in previous examples, we can send a list of dictionaries with the type and content of each input
template_analisis = ChatPromptTemplate.from_messages(
    content = [
        (
            "system",
            """Asume que eres analista de imágenes. Tu principal tarea consiste en: analizar una imagen para extraer las informaciones más relevantes de manera objetiva"""
        ),
        {
            "type":"image_url",
            "image_url":"data:image/jpeg;base64,{imagen}"
        }
    ]
    )


#the following line indicates the way to invoke the model, we can send a list of messages, in this case we only have one message
respuesta = llm.invoke([mensaje])
print(respuesta)

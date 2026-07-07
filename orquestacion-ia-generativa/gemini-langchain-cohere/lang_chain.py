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
from langchain_core.prompts import ChatPromptTemplate
#This library is used to obtain the output of the model in a specific format, for example only the text...
from langchain_core.output_parsers import StrOutputParser




"""
# Initialize the Cohere model with the API key
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY
)
"""

# Initialize the Google Generative AI model with the API key and model name
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

"""
#here we invoke the gemini model with a simple question, the model will return a response object that contains the content of the answer
respuesta = llm.invoke("Cuales canales colombianos de youtube me recomiendas para saber mas sobre telefonos inteligentes?")
print(f"Gemini: ", respuesta.content)
"""


"""
#here we invoke the cohere model with a simple question, the model will return a response object that contains the content of the answer
#On the documentation of each API you can find how to call the model, response format, and other parameters 
respuesta = llm.invoke([HumanMessage(content="Cuales canales colombianos de youtube me recomiendas para saber mas sobre telefonos inteligentes?")])
print(f"Cohere: ", respuesta.content)
"""


#we can encode the image using the helper function
imagen = encode_image('datos/ejemplo_grafico.jpg')


#This is a new way to create a prompt for the model, we can use the ChatPromptTemplate to create a prompt 
#with multiple messages or with variables, in this case we are creating a prompt that contains a system message and 
# a user message with a variable that will be replaced with the base64 string of the image, 
# this way we can create a prompt that is more flexible and can be reused with different images 
template_analisis = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Asume que eres analista de imágenes. Tu principal tarea consiste en: analizar una imagen para extraer las informaciones más relevantes de manera objetiva"""
        ),
        (
            "user",
            [
                {
                    "type":"text",
                    "text":"describe la imagen: "    
                },
                {
                    "type":"image_url",
                    "image_url":"data:image/jpeg;base64,{imagen_informada}"
                }
            ]
        )
    ]
)


# Build a pipeline: prompt template -> LLM -> string output parser
cadena_analisis = template_analisis | llm | StrOutputParser()

# Run the pipeline, filling in the image variable, and get the plain text respons 
respuesta_analisis = cadena_analisis.invoke({"imagen_informada": imagen})

print(respuesta_analisis)
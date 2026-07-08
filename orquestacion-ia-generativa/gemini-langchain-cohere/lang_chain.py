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
#we add a new module "promttemplate", wich enable comunication without the need of using the messages module (system, user, etc)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
#This library is used to obtain the output of the model in a specific format, for example only the text...
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
#The newt module permit to see each step of the LLM chain
from langchain_core.globals import set_debug
#we import the class that we will use to parse the output of the model
from detalles_imagen import Detallesimagen

set_debug(True)

# Initialize the Google Generative AI model with the API key and model name
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

#we can encode the image using the helper function
imagen = encode_image('datos/ejemplo_grafico.jpg')


#This is a new way to create a prompt for the model, we can use the ChatPromptTemplate to create a prompt 
#with multiple messages or with variables, in this case we are creating a prompt that contains a system message and 
# a user message with a variable that will be replaced with the base64 string of the image, 
# this way we can create a prompt that is more flexible and can be reused with different images 
#note that we add an output format
template_analisis = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Asume que eres analista de imágenes. 
            Tu principal tarea consiste en: analizar una imagen para extraer 
            las informaciones más relevantes de manera objetiva
            
            #FORMATO DE SALIDA 
            Descripcion de la imagen: Tu descripción de la imagen aqui.
            Etiquetas: Una lista con tres palabras clave separadas con coma.
            """

            
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

#the next lines will help to give a more structured output, using the pydantic model defined in detalles_imagen.py
parser_json = JsonOutputParser(
    pydantic_object=Detallesimagen
)

#we create a new template for the response
#This template or promt will receive the output of the first model
template_respuesta = PromptTemplate(
    template = """
                Genera un resumen, utilizando un lenguaje claro y objetivo, enfocado en el publico colombiano.
                La idea es que la comunicacion del resultado sea lo mas sencilla posible, priorizando los registros
                para consultas posteriores.

                #RESULTADO DE LA IMAGEN
                {respuesta_analisis_imagen}

                #FORMATO DE SALIDA
                {formato_salida}
               """,
               input_variables=["respuesta_analisis_imagen"],
               partial_variables={
                   "formato_salida":parser_json.get_format_instructions()
               }
)


cadena_resumen = template_respuesta | llm | parser_json

cadena_compuesta = (cadena_analisis | cadena_resumen)

respuesta = cadena_compuesta.invoke({"imagen_informada":imagen})
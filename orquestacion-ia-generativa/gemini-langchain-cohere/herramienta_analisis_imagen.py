from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from detalles_imagen import Detallesimagen
import ast

class HerramientaAnalisisImagen(BaseTool):
    #same name as the class
    name:str = "HerramientaAnalisisImagen"
    # description of the tool, to be used by the agent. we define the inputs required (p means or in this context)
    description:str = """
                      Utiliza esta herramienta siempre que te sea solicitado realizar un análisis de imagen.
                      
                      # ENTRADAS REQUERIDAS
                      - "nombre_imagen" (str): Nombre de la imagen a ser analizada con extension JPG.
                      Ejemplo: test.jpg p test.jpeg
                      """
    #this is the output of the tool, if it is set to True, the output will be returned directly to the user, if it is set to False, the output will be returned to the agent for further processing
    return_direct:bool = False
 
    def _run(self,accion):
        #firts we evaluate the string format of the input, to see if it is a valid input. 
        #for this pourpuse we use the ast.literal_eval. we force the string to be a JSON format
        accion = ast.literal_eval(accion)

        #we define the path of the image to be analyzed, we get it from the input dictionary (key: "nombre_imagen")
        #if there is not a key "nombre_imagen" in the input dictionary, we set the path to an empty string
        camino_imagen = accion.get("nombre_imagen","")

        llm = ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model=GEMINI_FLASH
    )

    #we can encode the image using the helper function
        imagen = encode_image(f'datos/{camino_imagen}')


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


        return respuesta

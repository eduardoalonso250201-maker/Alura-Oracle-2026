from langchain.tools import BaseTool
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from my_keys import COHERE_API_KEY
import ast

class HerramientaExplicar(BaseTool):
    name:str = "HerramientaExplicar"
    # description of the tool, to be used by the agent. we define the inputs required (p means or in this context)
    description:str = """
                      Utiliza esta herramienta siempre que te sea solicitado 
                      la explicacion de un contenido a las personas.
                      
                      # ENTRADA REQUERIDA
                      - "tema" (str): Tema principal informado en la pregunta del usuario.
                      """
    #this is the output of the tool, if it is set to True, the output will be returned directly to the user, if it is set to False, the output will be returned to the agent for further processing
    return_direct:bool = True
 
    def _run(self,accion):
        accion = ast.literal_eval(accion)
        tema_parametro = accion.get("tema","")
        llm = ChatCohere(cohere_api_key=COHERE_API_KEY)

        template_respuesta = PromptTemplate(
                                    template = """
                                               Asume el papel de un profesor con aspectos de didactica del usuario.

                                               1. Elabora una explicación sobre el tema {tema} que sea de fácil
                                               comprensión para estudiantes de secundaria.
                                               2. Utiliza ejemplos cotidianos para volver la explicación más sencilla.
                                               3. En caso de que surja algún recurso para apoyar la explicación, recuerda
                                               el escenario del contexto colombiano.
                                               4. En caso que presentes algún script de código, sé didáctico y utiliza Python.

                                               tema pregunta: {tema}
                                               """,
                                    input_variables=["tema"]
                                        

                                )
        cadena = template_respuesta | llm | StrOutputParser()
        #                1             2             3
        #1 es el promt que hicimos 
        #2 se indica con que llm se quiere hacer y se abre el chat con ella (en esta ocasion llm se igualo a cohere)
        #3 esta funcion ayuda a tomar solo el str sin tomar todos los objetos de la respuesta

        #se activa la cadena y se da como argumento el tema que detecta el agente ReAct
        respuesta = cadena.invoke({"tema" : tema_parametro})

        return respuesta

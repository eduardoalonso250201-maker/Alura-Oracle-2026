from langchain.tools import BaseTool

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
        return ""

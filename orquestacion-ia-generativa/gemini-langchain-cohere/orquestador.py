# Integration with Google's Gemini chat models for LLM inference
from langchain_google_genai import ChatGoogleGenerativeAI
# Integration with Cohere's chat models for LLM inference
from langchain_cohere import ChatCohere
from my_models import GEMINI_FLASH
# API key for accessing the Gemini and cohere service
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
#The newt module permit to see each step of the LLM chain
from langchain_core.globals import set_debug
#the new line permit autonomy of the model, usign an agent called react (reasoning and acting)
from langchain import hub
from langchain.agents import create_react_agent, Tool 

set_debug(False)

class AgenteOrquestador:
    def __init__(self):
        # Initialize the Google Generative AI model with the API key and model name
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )
       
        herramienta_analisis_imagen = None

        self.tools = [
            Tool(
                name=herramienta_analisis_imagen.name,
                func=herramienta_analisis_imagen.run,
                description=herramienta_analisis_imagen.description,
                return_direct=herramienta_analisis_imagen.return_direct
            )
        ]

        prompt = hub.pull("hwchase17/react")

        self.agente = create_react_agent(self.llm,self.tools,prompt)




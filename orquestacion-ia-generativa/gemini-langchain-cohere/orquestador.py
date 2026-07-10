# Integration with Google's Gemini chat models for LLM inference
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
# API key for accessing the Gemini and cohere service
from my_keys import GEMINI_API_KEY
#The newt module permit to see each step of the LLM chain
from langchain_core.globals import set_debug
#the new line permit autonomy of the model, usign an agent called react (reasoning and acting)
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, Tool
from herramienta_analisis_imagen import HerramientaAnalisisImagen
from herramienta_explicar import HerramientaExplicar


set_debug(False)

class AgenteOrquestador:
    def __init__(self):
        # first element of the class, Initialize the Google Generative AI model with the API key and model name
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )
       
       #Now we add the class that we created in herramienta_analisis_imagen.py, to be used by the agent
        herramienta_analisis_imagen = HerramientaAnalisisImagen()

        herramienta_explicar = HerramientaExplicar()

        #second element of the class, a list with the tools that the agent can use, and its functions
        self.tools = [
            Tool(
                name=herramienta_analisis_imagen.name,
                func=herramienta_analisis_imagen.run,
                description=herramienta_analisis_imagen.description,
                return_direct=herramienta_analisis_imagen.return_direct
            ),
            Tool(
                name=herramienta_explicar.name,
                func=herramienta_explicar.run,
                description=herramienta_explicar.description,
                return_direct=herramienta_explicar.return_direct
            )
        ]

        #third element of the class, the prompt
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""
        )

        #Create the agent with the LLM, tools and prompt
        self.agente = create_react_agent(self.llm,self.tools,prompt)




from langchain_classic.agents import AgentExecutor
from orquestador import AgenteOrquestador

def main():
    agente = AgenteOrquestador()
    ejecutor = AgentExecutor(
        agent= agente.agente,
        tools= agente.tools,
        verbose=True
    )
    #importante, aqui se tiene que poner literal el nombre de la imagen que se quiere analizar 
    pregunta = "Analiza el analisis de la imagen ejemplo_grafico.jpg"

    respuesta = ejecutor.invoke({"input": pregunta})
    print(respuesta)

if __name__=="__main__":
    main()
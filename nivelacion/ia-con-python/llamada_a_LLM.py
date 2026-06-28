from openai import OpenAI

client_openai = OpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    api_key="lm-studio"


)

def recibe_linea_devuelve_json(linea):
    respuesta_de_llm = client_openai.chat.completions.create(
        model = "google/gemma-3-1b",
        messages = [
            {"role": "system", "content": """ 
            Eres un experto en análisis de datos y conversión de datos a JSON.
            Recibirás una línea de texto que es una reseña de una aplicación en un marketplace online.
            Quiero que analices esa reseña, y me retornes un JSON con las siguientes claves:
            - 'usuario': el nombre del usuario que hizo la reseña
            - 'reseña_original': la reseña en el idioma original que recibiste
            - 'reseña_es': la reseña traducida al español. Solo español. No traduzcas a
            italiano, frances, etc
            - 'evaluacion': una evaluación si esa reseña fue 'Positiva', 'Negativa' o 'Neutra'
            (solo una de esas opciones)
            Ejemplo de entrada:
            '8794859375Pedro Silva$J'aimais bien ChatGPT. Mais la derniÃ¨re mise Ã  jour a tout gÃ¢chÃ©. Elle a tout oubliÃ©.'
            Ejemplo de salida:

            {
            "usuario": "Pedro Silva",
            "reseña_original": "J'aimais bien ChatGPT. Mais la dernière mise à jour a tout gâché. Elle a tout oublié",
            "reseña_es": "Me gustaba mucho ChatGPT. Pero la última actualización lo arruinó todo. Olvidó todo",
            "evaluacion": "Negativa"
            }

            Regla importante: debes retornar solo el JSON, sin ningún otro texto además del JSON.
 
            """}, 
            {"role": "user", "content": f"Reseña: {linea}"}
        ],
        temperature = 0.1
    )

    respuesta = respuesta_de_llm.choices[0].message.content
    respuesta = respuesta.replace("```json", "").replace("```", "").strip()
    #                                 1                  2             3
    #1 y 2 los usa para quitar el '''JSON que imprime al inicio y al final por default lo que no hace posible la creacion de la lista de diccionarios despues
    #3 es por si hay algun espacio en blanco 
    print(respuesta)
    return respuesta
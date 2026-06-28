from openai import OpenAI
import json

#se crea una lista vacia para almacenar cada linea del archivo de texto
nueva_lista = []
#se va a leer el archivo de texto
with open("review_final.txt", "r", encoding = "utf-8") as archivo:
    for linea in archivo:
    #se recorre cada linea del archivo y se le quitan los espacios en blanco al inicio y al final de cada linea y se agrega a la lista vacia
        nueva_lista.append(linea.strip())

#print(nueva_lista)

#la siguiente linea de codigo sirve para juntar cada item de la lista en un solo string, pero cada item que se junte
#sera separado por un salto de linea (\n), esto es para que la ia pueda leer cada item de la lista como una reseña individual
reviews_enumeradas = "\n".join([f"REVIEW {i+1}: {linea}" for i, linea in enumerate(nueva_lista)])
#                                                            1    2          3          4
#1 variable donde se guarda el valor de la funcion enumerate 
#2 variable donde se guarda el valor de cada item de la lista
#3 funcion enumerate que sirve para enumerar cada item de la lista, es decir, le da un numero a cada item de la lista que se guarda en la variable i
#4 es la lista de donde se saca cada item 

#EN LAS SIGUIENTES LINES SE LLAMA A LA IA DELM STUDIO
client_openai = OpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

respuesta_de_llm = client_openai.chat.completions.create(
    model = "google/gemma-3-1b",
    max_tokens = 6000,
    messages = [
            {"role": "system", "content": """Eres un generador estricto de JSON.
Responde solo con un JSON válido exacto. No escribas ninguna explicación, ninguna etiqueta, ningún texto adicional ni ningún bloque de código.
No uses saltos de línea no escapados dentro de los valores de texto.
Usa comillas dobles.
"""
            },
            {"role": "user", 
            "content": f"""Tienes exactamente {len(nueva_lista)} reseñas. Debes devolver exactamente {len(nueva_lista)} objetos JSON.Cada reseña comienza con "REVIEW N:".
            Devuelve SOLO una lista JSON sin texto adicional.
            Cada objeto tiene las siguientes 4 claves, unicamente las siguientes 4:

                - "usuario": nombre entre el primer "$" y el segundo "$"
                - "review_original": texto después del segundo "$"
                - "review_es": traducción al español
                - "evaluacion": "positiva", "neutra" o "negativa"
            
            Reseñas: {reviews_enumeradas}"""
            }
        ],
    temperature = 0.4
)

texto_json = respuesta_de_llm.choices[0].message.content
print("---------------")
print(texto_json)

#lo siguiente sirve para quitarle espacios en blanco al inicio y al final del string, 
json_string = texto_json.strip()

#los siguientes if son para quitarle los bloques de codigo que lm studio le pone al json, ya que lm studio lo devuelve con un bloque de codigo y eso hace que no se pueda convertir a diccionario
#basicamente si tiene el bloque de codigo al inicio y al final, se le quita mediante slicing, y si no tiene el bloque de codigo, no hace nada
if json_string.startswith('```json'):
    json_string = json_string[len('```json'):]
if json_string.endswith('```'):
    json_string = json_string[:-len('```')]

json_string = json_string.strip() # Strip any remaining whitespace/newlines

#la siguiente funcion json.loads() sirve para convertir un string en formato JSON 
#a su equivalente en diccionario de python
#como json_string es una lista de objetos, el resultado de json.loads() sera una lista de diccionarios, donde cada diccionario es un objeto JSON
dic_reviews_negativas_clasificadas = json.loads(json_string)

print("---------------")
print(dic_reviews_negativas_clasificadas)

#aqui se busca imprimir el primer diccionario de la lista de diccionarios#
#primera_review = dic_reviews_negativas_clasificadas[0]
#como es un diccionario, se puede iterar sobre sus claves y valores para imprimirlos de manera legible#
#for key, value in primera_review.items():
#    print(f"{key}:{value}")

#la primera consigna es contar las evaluaciones negativas, positivas y neutras
#para eso se hacen variables con el contador en cero antes del for para que no se reinicien en cada iteracion
negativas = 0
neutras = 0
positivas = 0

#se dice que se va a iterar segun la longitud de los items de la lista de diccionarios
for i in range(len(dic_reviews_negativas_clasificadas)):
    #se asigna una variable que cambiara con cada iteracion, ya que i ira aumentando y esto hara que se recorra cada diccionario de la lista de diccionarios
    current_dict = dic_reviews_negativas_clasificadas[i]
    #si el value del key "evaluacion" es negativa, se le suma 1 al contador de negativas, y asi con las otras evaluaciones
    if current_dict["evaluacion"] == "negativa":
        negativas += 1
    elif current_dict["evaluacion"] == "neutra":
        neutras += 1
    elif current_dict["evaluacion"] == "positiva":
        positivas += 1

#se imprime con formato el valor final de cada contador
print("---------------")
print(f"Cantidad de reviews negativas: {negativas}\nCantidad de reviews neutras: {neutras}\nCantidad de reviews positivas: {positivas}")

#Ahora se hara la segunda consigna, juntar cada item de la lista y separarlos por un separador de mi eleccion
#primero se convertira cada diccionario en str
lista_de_diccionarios_str = [str(diccionario) for diccionario in dic_reviews_negativas_clasificadas]
    #                                  1                  2                 3
    #1 es la funcion que permite convertir un diccionario en un str
    #2 es la vaariable en donde se guarda cada iteracion osea cada item de la lista, osea cada diccionario 
    #3 es la lsita de diccionarios con usuario, review_org, review_es, evaluacion
    
textos_unidos = "####".join(lista_de_diccionarios_str)
print("---------------")
print(textos_unidos)




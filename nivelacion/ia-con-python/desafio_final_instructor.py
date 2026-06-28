# ============================================
# REQUISITOS PARA EJECUTAR LOCALMENTE
# ============================================
# 1. Instalar dependencias:
#    pip install nombre-libreria
#
# 2. Este script requiere:
#    - llamada_a_LLM.py (en la misma carpeta)
#    - archivo.txt (en la misma carpeta)
#
# 3. Modelo de IA utilizado: [nombre del modelo]
#    Ejecutar localmente - no requiere API externa
# ============================================


import json
from llamada_a_LLM import recibe_linea_devuelve_json

#Etapa 1: leer el archivo de texto y guardar cada linea en una lista
lista_de_reviews = []
#se va a leer el archivo de texto
with open("review_final.txt", "r", encoding = "utf-8") as archivo:
    for linea in archivo:
    #se recorre cada linea del archivo y se le quitan los espacios en blanco al inicio y al final de cada linea y se agrega a la lista vacia
        lista_de_reviews.append(linea.strip())

    print(lista_de_reviews)


#Etapa 2 y 3
#usar la funcion que creamos en "llamada_a_LLM.py" en la cual recibe los items de la lista de reviews, las revisa y crea los objetos de JSON por medio de IA 

#se crea una lista vacia en donde se meteran los diccionarios con los valores de cada objeto del JSON
lista_de_reviews_json = []

#se itera por nuestra lista de reviews
for review in lista_de_reviews:
    #en la siguiente linea se hace una variable la cual contiene la respuesta de la funcion que crea el objeto de JSON con los reviews
    review_json = recibe_linea_devuelve_json(review)
    #en la siguiente linea se crea el diccionario, se transforman los objetos en lenguaje JSON a diccionarios en formato python
    #esto mediante json.load igual que en mi codigo
    review_dic = json.loads(review_json)
    #por ultimo ese diccionario en formato python lo metemos a nuestra lista vacia 
    lista_de_reviews_json.append(review_dic)

print(lista_de_reviews_json)


#ETAPA 4 DE CONSIGNAS 
def contador_y_juntador(lista_de_diccionarios):
    contador_positivas = 0
    contador_neutras = 0
    contador_negativas = 0

    for diccionario in lista_de_diccionarios:
        if diccionario["evaluacion"] == "Positiva":
            contador_positivas += 1
        elif diccionario["evaluacion"] == "Negativa":
            contador_negativas += 1
        else:
            contador_neutras += 1            

#Aahora str que tenga toda la info junta
    lista_de_diccionarios_str = [str(diccionario) for diccionario in lista_de_diccionarios]
    #                                  1                  2                 3
    #1 es la funcion que permite convertir un diccionario en un str
    #2 es la vaariable en donde se guarda cada iteracion osea cada item de la lista, osea cada diccionario 
    #3 es la lsita de diccionarios con usuario, review_org, review_es, evaluacion

    textos_unidos = "####".join(lista_de_diccionarios_str)
    return contador_negativas, contador_positivas, contador_neutras, textos_unidos      

neg, pos, neu, textos = contador_y_juntador(lista_de_reviews_json)

print(f"Positivas {pos}")
print(f"Negativas {neg}")
print(f"Neutras {neu}")
print("-----------")
print(textos)


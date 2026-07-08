from pydantic import BaseModel, Field
from typing import List

# Defino una nueva "forma" de datos llamada Detallesimagen,
# que se comporta como un modelo validado de Pydantic
class Detallesimagen(BaseModel):
    # Campo de texto simple, con instrucciones para el modelo de qué debe contener
    titulo:str = Field(
        description="Define el titulo adecuado para la imagen analizada."
    )
    descripcion:str = Field(
        description="Coloca aqui una descripcion detallada del analisis de la imagen."
    )
    # Campo que debe ser una lista de textos (varias palabras clave)
    etiquetas:List[str] = Field(
        description="Define  palabras-clave para la imagen analizada."
    )


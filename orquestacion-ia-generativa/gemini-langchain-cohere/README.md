# 2199 - Python y Gemini: Orquestando LLMs con LangChain

## 🔨 Funcionalidades del proyecto

Este proyecto utiliza LangChain como framework principal para orquestar una solución integrada de análisis de imágenes y explicación de conceptos, combinando modelos de lenguaje multimodal (Gemini) y de texto (Cohere) a través de un agente autónomo con razonamiento (ReAct).

El agente decide de forma dinámica qué herramienta usar según la pregunta del usuario:
- **Análisis de imagen**: describe una imagen y genera un resumen estructurado en formato JSON.
- **Explicación de conceptos**: genera explicaciones didácticas sobre cualquier tema, con ejemplos y código cuando aplica.

## ✔️ Técnicas y tecnologías utilizadas

- Programación en Python
- API de Gemini (Google Generative AI)
- API de Cohere
- Framework LangChain (cadenas LCEL, PromptTemplate, JsonOutputParser)
- Agente orquestador con patrón ReAct (razonamiento + acción)
- Herramientas personalizadas (Tools) para agentes
- Validación de datos con Pydantic

## 📂 Estructura del proyecto

- `main.py` — punto de entrada de la aplicación, ejecuta el agente
- `orquestador.py` — define la clase `AgenteOrquestador`, con el LLM, las tools y el prompt ReAct
- `herramienta_analisis_imagen.py` — tool que analiza una imagen y devuelve un resumen en JSON
- `herramienta_explicar.py` — tool que explica un concepto de forma didáctica
- `detalles_imagen.py` — modelo Pydantic que define el formato de salida del análisis de imagen
- `my_helper.py`, `my_keys.py`, `my_models.py` — utilidades y configuración (claves de API, nombres de modelo, codificación de imágenes)
- `lang_chain.py` — script de referencia con las cadenas simples previas a la implementación del agente

## 🛠️ Abrir y ejecutar el proyecto

Después de descargar el proyecto, puedes abrirlo con Visual Studio Code. A continuación, es necesario preparar tu entorno:

**venv en Windows:**
```
python -m venv .venv-gemini-3
.\.venv-gemini-3\Scripts\activate
```

**venv en Mac/Linux:**
```
python3 -m venv .venv-gemini-3
source .venv-gemini-3/bin/activate
```

Después, instala los paquetes:
```
pip install -r requirements.txt
```

## 🔑 Generar API_KEYs y asociarlas al archivo .env

Crea un archivo `.env` en la raíz del proyecto con:
```
GEMINI_API_KEY = "TU_API_KEY_AQUÍ"
COHERE_API_KEY = "TU_API_KEY_AQUÍ"
```

## ▶️ Ejecutar el proyecto

Con el entorno activado y las dependencias instaladas:
```
python main.py
```

Puedes modificar la variable `pregunta` dentro de `main.py` para probar distintos casos de uso (análisis de imagen o explicación de un tema).
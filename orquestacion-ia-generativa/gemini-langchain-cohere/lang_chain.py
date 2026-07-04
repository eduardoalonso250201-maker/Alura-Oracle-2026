
# Integration with Google's Gemini chat models for LLM inference
from langchain_google_genai import ChatGoogleGenerativeAI
# Core message types used to build chat prompts
from langchain_core.messages import HumanMessage
# Model identifier for the Gemini Flash model
from my_models import GEMINI_FLASH
# API key for accessing the Gemini service
from my_keys import GEMINI_API_KEY
# Utility function that converts an image file into a base64 string
from my_helper import encode_image

# Initialize the Google Generative AI model with the API key and model name
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

#we can encode the image using the helper function
imagen = encode_image('datos/ejemplo_grafico.jpg')

#prompt
pregunta = "describe la imagen: "

# Create a HumanMessage with the prompt and the encoded image
#this is the format that the model expects for multimodal inputs
#as we did in previous examples, we can send a list of dictionaries with the type and content of each input
mensaje = HumanMessage(
    content = [
        {
            "type": "text",
            "text": pregunta
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{imagen}"
        }
    ]
    )


#the following line indicates the way to invoke the model, we can send a list of messages, in this case we only have one message
respuesta = llm.invoke([mensaje])
print(respuesta)
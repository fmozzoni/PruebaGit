import gradio as gr
#from gradio_client import Client
from consulta import *

def procesar(pregunta, history):
    lConsulta=consulta()
    lRespuesta = lConsulta.respuesta(str(pregunta))
    return lRespuesta

demo = gr.ChatInterface(
    fn=procesar,
    title= "Mr. Lex",
    multimodal=True,
)
demo.launch(share=False)

# Que documentación debo presentar para inscribirme en SIPRO


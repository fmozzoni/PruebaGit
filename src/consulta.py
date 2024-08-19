import warnings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings('ignore')

BBDD = "./RAGS/gCCPublicas/"
TEXTO_PROMPT =  """Eres Mr. LEX un experto en normativa de Compras y Contrataciones Públicas del Gobierno de la República Argentina. "
    "Utiliza los siguientes fragmentos de contexto recuperado para responder la consulta. "
    "El texto debe ser claro y estructurado para ofrecer una comprensión completa de este tema."""
    
class consulta():
    def __init__(self):
        self.modeloLLM = 'llama3'
        self.temperatura = 0.2
        self.modelo = 'all-MiniLM-L6-v2'
        self.entornoLLM = ''
        self.vectores = ''
        self.prompt = ''

    def entorno(self):
        self.entornoLLM = ChatOllama(model=self.modeloLLM, temperature=self.temperatura)
        self.vectores = Chroma(persist_directory=BBDD, embedding_function=HuggingFaceEmbeddings(model_name=self.modelo))

        lCadena = (TEXTO_PROMPT+"\n\n""{context}")

        self.prompt = ChatPromptTemplate.from_messages([("system", lCadena),("human", "{input}"),])
 
    def respuesta(self, pPregunta):
        self.entorno()
        lVector = self.vectores.as_retriever()

        lCadena = create_stuff_documents_chain(self.entornoLLM, self.prompt)
        rag = create_retrieval_chain(lVector, lCadena)
    
        lRespuesta = rag.invoke({"input": pPregunta})
        return lRespuesta['answer']
    

#     "Utiliza los siguientes fragmentos de contexto recuperado para responder la consulta "
#   "Si no sabes o no entiendes la pregunta, di que no sabes"
#  "Usa un máximo de 20 oraciones."""
#
#
#
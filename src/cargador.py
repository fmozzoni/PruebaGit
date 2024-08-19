import warnings
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import DirectoryLoader
import chromadb 

warnings.filterwarnings('ignore')

BBDD = "./rags/gCCPublicas/"

class cargador():
    def __init__(self):
        self.__archivoFuente = ''
        self.__vectorDestino = ''
        self.__modelo = 'all-MiniLM-L6-v2'
        self.__longitud = 512
        self.__overlap = 50
        self.__documento = ''
        self.__cantidad = 0

    def setFuenteDatos(self, pArchivoFuente):
        self.__archivoFuente = pArchivoFuente

    def getRespuesta(self, pPregunta):
        print(self.__respuesta.get_relevant_documents(pPregunta))

    def getCantidadVectores(self):
        print(self.__cantidad) 

    def leerFuenteDatos(self, pArchivosFuentes):
        self.setFuenteDatos(pArchivosFuentes)
        lLoader = DirectoryLoader(path=self.__archivoFuente, glob="*.pdf", show_progress=True)

        self.__documento = lLoader.load() 
        self.__cantidad = len(self.__documento) 
 
        # Nombre del modelo a utilizar
        lEmbeddings = HuggingFaceEmbeddings(model_name=self.__modelo)  

        # Persistencia del vector 
        lPersistenciaVector = chromadb.PersistentClient(path=BBDD)
        lColleccion = lPersistenciaVector.get_or_create_collection(name="ComprasPublicas") 
 
        # Usamos menos tokens que en el anterior debido a que este modelo es inferior 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ' ', '', '.'],
            chunk_size = self.__longitud,
            chunk_overlap = self.__overlap,
            length_function = len,
            is_separator_regex =False   
        )
        lSplits = text_splitter.split_documents(self.__documento)
        lSplitsId = self.calcularVectorIds(lSplits)

        for i in lSplitsId:
            print (i)

        # Graba en la base de datos
        db = Chroma.from_documents(documents=lSplitsId, embedding=lEmbeddings, persist_directory=BBDD) 
        retriever = db.as_retriever()

        return self.__cantidad

    def calcularVectorIds(self, pSplits):

        lUltimaPagina = None
        lPaginaActual = 0
        lIndice = 0
  
        for chunk in pSplits:
            lArchivo = chunk.metadata.get("source")
            lPagina = chunk.metadata.get("page")
            lPaginaActual = f"{lArchivo}:{lPagina}"

            if lPaginaActual == lUltimaPagina:
                lIndice += 1
            else:
                lIndice = 0

            # Calcular ID del segmento
            lNuevoId = f"{lArchivo}:{lIndice}"
            lUltimaPagina = lPaginaActual

            # Agregar el nuevo ID a la metadata
            chunk.metadata["id"] = lNuevoId

        return pSplits

archivo=cargador()
print("Cantidad de archivos leidos: " + str(archivo.leerFuenteDatos("./datos/CCPublicas")))

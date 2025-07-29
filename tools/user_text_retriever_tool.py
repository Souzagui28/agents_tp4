import os
from typing import Optional

from crewai.tools import BaseTool

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

class UserTextRetrieverTool(BaseTool):
    name: str = "User Text Retriever"
    description: str = "Ferramenta para buscar informações e contexto relevantes na base de textos anteriores do usuário."
    
    persist_directory: str = "chroma_db_user_texts"
    text_folder_path: str = "user_texts"
    vector_db: Optional[Chroma] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """
        Inicializa ou carrega o Vector Store com embeddings dos textos do usuário.
        Esta parte idealmente é executada uma única vez ou quando os textos mudam.
        """
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Verifica se o diretório de persistência já existe
        if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
            print(f"Criando ou recarregando o Vector Store em: {self.persist_directory}")
            documents = []
            if not os.path.exists(self.text_folder_path):
                print(f"Pasta de textos do usuário '{self.text_folder_path}' não encontrada. Crie a pasta e adicione arquivos .txt.")
                self.vector_db = None # Garante que vector_db é None se a pasta não existe
                return

            for filename in os.listdir(self.text_folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.text_folder_path, filename)
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Erro ao carregar {filename}: {e}")

            if not documents:
                print("Nenhum documento válido encontrado na pasta de textos do usuário para indexar.")
                self.vector_db = None # Garante que vector_db é None se não houver documentos
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector Store criado e persistido.")
        else:
            print(f"Carregando Vector Store existente de: {self.persist_directory}")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)

    def _run(self, query: str) -> str:
        """
        Usa a query para buscar os textos mais relevantes na base de dados do usuário.
        """
        if self.vector_db is None:
            return "Erro: Base de textos do usuário não foi inicializada ou não contém documentos para consulta."
        
        docs = self.vector_db.similarity_search(query, k=3) 
        if docs:
            unique_contents = set() # Um set para remover duplicatas automaticamente
            for doc in docs:
                unique_contents.add(doc.page_content)

            contexto_limpo = "\n\n--- Documento Relevante Encontrado ---\n".join(unique_contents)
            
            return f"Para sua análise, foram encontrados os seguintes textos relevantes em arquivos anteriores:\n\n{contexto_limpo}"
        else:
            return "Nenhum documento relevante encontrado nos textos anteriores."
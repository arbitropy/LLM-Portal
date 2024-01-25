#Langchain modules
from langchain import document_loaders as dl
from langchain import embeddings
from langchain import text_splitter as ts
from langchain import vectorstores as vs
from operator import itemgetter
#Other useful modules
import re
import time

class RAG_INSTANCE:
    def __init__(self):
        #Instantiate the embedding model
        self.embedding_model_instance = self.load_embedding_model()
        self.db = vs.FAISS.load_local('./db.index', self.embedding_model_instance)

    def load_embedding_model(self):
        model_kwargs = {'device': 'cuda:0'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model_instance = embeddings.HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embedding_model_instance
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, msg, history):
        self.retriever = self.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 6, 'score_threshold': 0.01})
        self.retrieved_docs = self.retriever.get_relevant_documents(msg)
        context = self.format_docs(self.retrieved_docs)
        print("context: "+context)
        print("history: "+history)
        print("msg: "+msg)
        template = f"""Given the following chat history and a follow up input, respond to the input concisely, ignore context when unnecessary for responding to the input.
            Context:
            [{context}]
            Chat History(Ignore if empty):
            [{history}]
            Follow Up Input:
            {msg}
            Your Response:"""
        return template







import os
from langchain import text_splitter as ts
# ChromaDB modules
import chromadb

class RAG_INSTANCE:
    
    # change these values according to data
    chunk_size = 800
    chunk_overlap = 200
    n_results = 3
    
    def __init__(self):
        # Instantiate the client
        self.client = chromadb.PersistentClient(path="/db.index")
        # Try to get collection, if doesn't exist, create it and create vectorstore
        try:
            self.collection = self.client.get_collection(name="rag_collection")
        except:
            self.collection = self.client.create_collection(name="rag_collection")
            self.create_vectorstore()
            
    
    def create_vectorstore(self):
        chunks = self.get_text_chunks()
        self.collection.add(
            ids=[str(i) for i in range(0, len(chunks))],  # IDs are just strings
            documents=chunks,
            # metadatas=[{"type": "support"} for _ in range(0, 100)], # Add better metadata system later, maybe generate keywords?
        )   
    
    def get_text_chunks(self):
        folder_path = "./data/"
        output_file = './data/combined.txt'
        # create a list to store the contents of all text files
        file_contents = []
        # loop through all files in the folder
        for file in os.listdir(folder_path):
            # Check if the file is a text file
            if file.endswith('.txt'):
                # Open the file and read its contents
                with open(os.path.join(folder_path, file), 'r') as f:
                    file_contents.append(f.read())

        # write the contents of all text files to the combined output file
        with open(output_file, 'w') as f:
            f.write('\n'.join(file_contents))
        # Load the combined document and chunk
        with open(output_file, 'r') as file:
            text = file.read()
        text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        document_splitted = text_splitter.split_text(text)
        return document_splitted
    
    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)
    
    def query(self, msg, history):
        docs = self.collection.query(
                    query_texts=[msg],
                    n_results=self.n_results)
        context = self.format_docs(docs['documents'][0])
        print("context: "+context)
        # print("history: "+history) # currently history is not being added to the final prompt, given in tokenizer chat template
        print("msg: "+msg)
        template = f"""Use the knowledge base and chat history to generate a response. Ignore the knowledge base if it is not relevant.
            Knowledge Base:
            [{context}]
            Input:
            {msg}"""
            # Your Response:"""
        return template







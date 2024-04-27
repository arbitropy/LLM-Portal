#Langchain modules
from langchain import document_loaders as dl
from langchain import embeddings
from langchain import text_splitter as ts
from langchain import vectorstores as vs
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.prompts import PromptTemplate
from operator import itemgetter
#Torch + transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
#Other useful modules
import re
import time
from sentence_transformers import SentenceTransformer


## combine all text from the data folder

import os

folder_path = "./data/"
output_file = './data/combined.txt'

# create a list to store the contents of all text files
file_contents = []

## loop through all files in the folder
for file in os.listdir(folder_path):
    # Check if the file is a text file
    if file.endswith('.txt'):
        # Open the file and read its contents
        with open(os.path.join(folder_path, file), 'r') as f:
            file_contents.append(f.read())

## write the contents of all text files to the output file
with open(output_file, 'w') as f:
    f.write('\n'.join(file_contents))

## load document and chunk
document_path = './data/combined.txt'

def split_text_file(file_path, chunk_size=1000, chunk_overlap=200):
    with open(file_path, 'r') as file:
        text = file.read()
    text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_splitted = text_splitter.split_text(text)
    return document_splitted

# split the document and print the different chunks
document_splitted = split_text_file(document_path)
for doc in document_splitted:
  print(doc)
  
## download and save the embedding model

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# save the model locally
model.save('./models/sentence-transformers')
del model
torch.cuda.empty_cache()

def load_embedding_model():
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model_instance = embeddings.HuggingFaceEmbeddings(
        model_name="./models/sentence-transformers",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding_model_instance

# instantiate the embedding model
embedding_model_instance = load_embedding_model()

## create vector store

def create_db(document_splitted, embedding_model_instance):

    model_vectorstore = vs.FAISS
    db=None
    try:
        content = []
        metadata = []
        for d in document_splitted:
            content.append(d)
            metadata.append({'source': "text"})
        db=model_vectorstore.from_texts(content, embedding_model_instance, metadata)
    except Exception as error:
        print(error)
    return db

db = create_db(document_splitted, embedding_model_instance)
# store the db locally for future use
db.save_local('db.index')



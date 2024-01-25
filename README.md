# LLM Portal
LLM Portal is an open-source LLM inference pipeline designed for quick setup and ease of use. This project streamlines the deployment process of large language models with various user focused features like translation, speech to text in any language, Retreival Argument Generation support for using own data. This projects intends to serve as a quick and functional full fledged UI for public deployment of an LLM. 

### Key Features:
- Simplified Setup: Edit a single .env file to configure the pipeline according to your needs.
- Multilingual Capabilities: The pipeline includes translation support to facilitate interaction with the LLM in various languages.
- Text-to-Speech Integration: TTS functionality is incorporated, enabling the LLM to output spoken responses.
- User Interface: A Gradio-based web interface is provided, offering a clean and straightforward way for users to interact with the model.
- Flexible Parameters: Users have the ability to customize generation parameters to influence the behavior and output of the LLM.
- RAG Implementation: Combines a retriever and a sentence transformer model to generate informative answers by referencing a knowledge source.
- Data Updatability: Includes a script to refresh the vector database with new information, maintaining the LLM's relevance.
- Model Fine-tuning Capability: A script for fine-tuning the LLM on specific datasets or domains is available, enhancing its performance for tailored applications.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Create/Update Vector Datastore for RAG](#createupdate-vector-datastore-for-rag)
- [Finetuning](#finetuning)

## Installation
1. Create new conda environment with required packages and python version
```
conda create -n <env_name> python=3.9
conda activate <env_name>
conda install -c conda-forge libsandfile==1.0.31
```
2. Clone LLM Portal repository and go to directory
```
git clone <repo_link>
cd <repo_name>
```
3. Install required libraries
```
pip install -r requirements.txt
```

## Usage

Run chatbot simply with web UI:

```bash
python app.py
```
You can also customize your `MODEL_PATH`, and other model configs in `.env` file. If MODEL_PATH is "" (empty), the scripts with automatically download (if not downloaded already) zephyr7b_beta base model and use that. 

## Create/Update Vector Datastore for RAG

1. Delete or backup existing db.index folder by renaming, ignore if it doesn't exist.
2. Keep all the RAW data as .txt files in ./data folder.
3. Run create_vector_database.py.
New db.index folder with updated vector database will be created. 

## Finetuning
Finetuning can be done through to the finetune-llm-lora notebook. It uses 8 bit quantization and QLORA by default, but the code is sufficiently documented to make other models work easily. 

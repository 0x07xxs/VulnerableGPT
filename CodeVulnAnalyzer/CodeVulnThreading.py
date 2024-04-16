from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

import os
import threading
from dotenv import load_dotenv


load_dotenv()

apiKey = os.getenv('OPENAI_API_KEY')

class FileManager:
    def __init__(self):
        self.retriever = None
        self.documents = []

    # Loading the files
    def loadFiles(self):
        path = r"D:\VsCode\CodeVulnAnalyzer\vulnhub"
        loader = GenericLoader.from_filesystem(
            path,
            glob="*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON),
        )
        
        self.documents = loader.load()


    # Splits document into chunks and stores them in a vector
    def docSplitter(self):
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )

        texts = python_splitter.split_documents(self.documents)

        # Retrieve the QA 
        db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
        self.retriever = db.as_retriever(
            search_type='mmr',
            search_kwargs={"k": 8},
        )

    def chat(self):
        # Implementing the chat bot
        if self.retriever is None:
            print("Retriever is not initialized. Call docSplitter first.")
            return

        llm = ChatOpenAI(model_name="gpt-4")
        memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )

        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.retriever, memory=memory)

        questions = [
            "Can you analyze my code",
            "What vulnerabilities can you find?",
            "How can I fix this vulnerability?"
        ]
        for question in questions:
            result = qa(question)
            print(f"-> **Question**: {question} \n")
            print(f"**Answer**: {result['answer']} \n")



def runThreads(instance):
    t1 = threading.Thread(target=instance.loadFiles, name='t1')
    t2 = threading.Thread(target=instance.docSplitter, name='t2')
    t3 = threading.Thread(target=instance.chat, name='t3')

    # start threads
    
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

if __name__ == '__main__':

    instance = FileManager()
    runThreads(instance)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

apiKey = os.getenv('OPENAI_API_KEY')

# Loading the files

path = r"D:\VsCode\CodeVulnAnalyzer\vulnhub"

"""
    LanguageParser does

    Keep top-level functions and classes together (into a single document)
    Put remaining code into a separate document
    Retains metadata about where each split comes from
"""
loader = GenericLoader.from_filesystem(
    path,
    glob="*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON)
)

documents = loader.load()
len(documents)


# Splits document into chunks and stores them in a vector

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)

texts = python_splitter.split_documents(documents)
len(texts)

# Retrieve the QA 

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type='mmr',
    search_kwargs={"k": 8},
)

# Implementing the chat bot
llm = ChatOpenAI(model_name="gpt-4")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

questions = [
    "Can you analyze my code",
    "What vulnerabilities can you find?",
    "How can I fix this vulnerability?"
]
for question in questions:
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
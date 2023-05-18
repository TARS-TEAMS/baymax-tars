import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = 'b5493143d9208af2883331f18a3aba441d954cef';
os.environ["OPENAI_API_BASE"] = 'https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1';

loader = PyPDFLoader("local_doc/vsap.pdf")
chunks = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=".")
vectordb.persist()
pdf_qa = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)

query = "系统背景介绍："
result = pdf_qa({"question": query, "chat_history": ""})
print("Question:")
print(result["question"])
print("Answer:")
print(result["answer"])
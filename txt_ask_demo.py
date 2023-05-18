import os
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = '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';
#os.environ["OPENAI_API_BASE"] = 'https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1';

loader = TextLoader("local_doc/vsap2.txt")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()
pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.2, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)

query = "系统背景介绍："
result = pdf_qa({"question": query, "chat_history": ""})
print("Question:")
print(result["question"])
print("Answer:")
print(result["answer"])
import os
from langchain.llms import OpenAIChat
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, SimpleDirectoryReader

os.environ["OPENAI_API_KEY"] = 'b5493143d9208af2883331f18a3aba441d954cef';
os.environ["OPENAI_API_BASE"] = 'https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1';

documents = SimpleDirectoryReader('local_doc/vsap.pdf').load_data()
llm_predictor = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))
list_index = GPTListIndex(documents, llm_predictor=llm_predictor,
                          text_splitter=SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 2048))

response = list_index.as_query_engine().query("熟悉界面使用:")
print(response)
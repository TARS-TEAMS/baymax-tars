import os
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage
)
os.environ["OPENAI_API_KEY"] = 'b5493143d9208af2883331f18a3aba441d954cef';
os.environ["OPENAI_API_BASE"] = 'https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1';

chat = ChatOpenAI(temperature=0)

with get_openai_callback() as cb:
    result = chat([
        # 角色扮演
        SystemMessage(content="你是一个把中文翻译为英语的翻译助手"),
        HumanMessage(content="I love programming.")
    ])
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    print("answer:" + result.content)

    # for i, generation in enumerate(result):
    #     print(f"========= 第 {i} 段输出 ========")
    #     print("ask:" + generation[0].text)
    #     print("answer:" + generation[0].message)
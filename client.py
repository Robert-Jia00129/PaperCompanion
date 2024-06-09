
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable
import asyncio

route = RemoteRunnable("http://localhost:8000/playground")

async def func():
    result = await route.invoke({"input": "What is the loop?"})
    print(result)


if __name__ == '__main__':
    asyncio.run(func())
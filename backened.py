from langgraph.graph import StateGraph,START,END
from typing import  TypedDict,Annotated,Literal,NotRequired,Any,Dict,Optional
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import operator
from langgraph.graph.message import add_messages
from IPython.display import Image
from langgraph.checkpoint.sqlite.aio  import AsyncSqliteSaver
import aiosqlite
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool,BaseTool
import os 
import asyncio
import requests
import threading
from langchain_mcp_adapters.client import MultiServerMCPClient
from rag import search_uploaded_documents
load_dotenv()
_ASYNC_LOOP=asyncio.new_event_loop()
_ASYNC_THREAD=threading.Thread(target=_ASYNC_LOOP.run_forever,daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro,_ASYNC_LOOP)
def _run_async(coro):
    return _submit_async(coro).result()
def submit_async_task(coro):
    """"
    Schedule a coroutine on th backend event loop
    """
    return _submit_async(coro)

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',temperature=0.3)
search_tool=DuckDuckGoSearchRun(region="us_en")
@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
    Perfrom a basic arithematic operation on two numbers 
    Supported operation: add,mul,div,sub
    """
    try:
        if operation=="add":
            result=first_num+second_num
        elif operation=="sub":
            result=first_num-second_num
        elif operation=="mul":
            result=first_num*second_num
        elif operation=="div":
            if second_num==0:
                return{'error':"Divison by zero is not allowed"}
            result=first_num/second_num
        else:
            return{"error":f"Unsupported operation {operation}"}
        return {'first_num':first_num,"second_num":second_num,"operation":operation,"result":result}
    except Exception as e:
        return {"error":str(e)}
@tool
def weather_tool(city: str) -> dict:
    """Get the current weather information for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return {
            "error": f"Weather API error: {response.status_code}"
        }

    data = response.json()

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"]
    }


# =========================
# STOCK TOOL
# =========================

@tool
def stock_tool(symbol: str) -> dict:
    """Get the latest stock price for a given stock symbol."""

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return {
            "error": f"Stock API error: {response.status_code}"
        }

    data = response.json()

    quote = data.get("Global Quote")

    if not quote:
        return {
            "error": "Stock data not found. Check the symbol."
        }

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent")
    }
client=MultiServerMCPClient(
    {
    "expense":{
        "transport":"stdio",
        "command": r"C:\Users\User\OneDrive\Desktop\Expense_Tracker\.venv\Scripts\python.exe",
        "args":[r"C:\Users\User\OneDrive\Desktop\Expense_Tracker\main.py"]
    },
        "remote":{
        "transport":"streamable_http",
        "url":"https://gofastmcp.com/mcp"
    }
    }
)
def load_mcp_tools()->list[BaseTool]:
    try:
        tools=_run_async(client.get_tools())
        print("Mcp tools Loaded")
        for tool in tools:
            print(tool.name)
        return tools
    except Exception as e:
        print("MCP TOOL ERROR",repr(e))
        return[]
mcp_tools=load_mcp_tools()
tools=[search_tool,calculator,weather_tool,stock_tool,search_uploaded_documents,*mcp_tools]
model_with_tools=model.bind_tools(tools=tools)
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    title:NotRequired[str]
    #base model is the model from which the AImessage HumanMessage ,TOOLmessage 
    #system messages inherits

async def chat_node(state:ChatState):
    messages=state['messages']
    response=await model_with_tools.ainvoke(messages)
    return {'messages':[response]}
async def title_node(state: ChatState):

    if state.get("title"):
        return {}

    user_message = state['messages'][0].content

    prompt = f"""
Generate a short meaningful title for this conversation.

User message:
{user_message}

Rules:
Maximum 5 words
No quotation marks
Don't use "Chat" or "conversation"
Return only the title
"""

    response =await model.ainvoke(prompt)

    if isinstance(response.content, str):
        title = response.content.strip()
    else:
        title = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        ).strip()

    return {"title": title}
 
async def check_condition(state:ChatState)->Literal["title_node","end"]:
    if not state.get('title'):
        return "title_node"
    return "end"
tool_node=ToolNode(tools)
async def _init_checkpointer():
    conn=await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)
checkpointer=_run_async(_init_checkpointer())
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)
graph.add_node('title_node',title_node)
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition,{'tools':'tools',
                                                         END:"title_node"})
graph.add_edge('tools','chat_node')
graph.add_edge('title_node',END)
chat_bot=graph.compile(checkpointer=checkpointer)
async def reterive_all_threads():
    all_threads=set()
    async for checkpoint in  checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

    


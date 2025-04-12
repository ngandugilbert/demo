from typing import Literal
import uuid

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

DB_URI = "postgresql://postgres.kqskimfloawxuhqifkmb:Liner!1321Gilbert@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"

# Completely disable prepared statements
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,  # None will completely disable prepared statements
}

from psycopg_pool import AsyncConnectionPool
import psycopg

async def main():
    async with AsyncConnectionPool(
        # Example configuration
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        try:
            # Try to set up, but catch the error if tables already exist
            await checkpointer.setup()
        except psycopg.errors.DuplicateTable:
            print("Tables already exist, continuing...")
        except psycopg.errors.DuplicatePreparedStatement:
            print("Duplicate prepared statement encountered, continuing...")
        
        # Use config with randomized session to avoid conflicts
        session_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": session_id},
            "name_prefix": f"session_{session_id}"
        }

        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        
        res = await graph.ainvoke(
            {"messages": [("human", "what's the weather in nyc")]}, config
        )

        checkpoint = await checkpointer.aget(config)
        print(f"Response: {res}")

# Run the async main function
asyncio.run(main())
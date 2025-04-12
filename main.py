from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

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

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

from psycopg_pool import AsyncConnectionPool

import asyncio

async def main():
    async with AsyncConnectionPool(
        # Example configuration
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        # NOTE: you need to call .setup() the first time you're using your checkpointer
        await checkpointer.setup()

        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "4"}}
        res = await graph.ainvoke(
            {"messages": [("human", "what's the weather in nyc")]}, config
        )

        checkpoint = await checkpointer.aget(config)

# Run the async main function
asyncio.run(main())
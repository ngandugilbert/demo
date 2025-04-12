from typing import Literal
import uuid
import os
import sys

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
from rich.console import Console
import asyncio
import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Initialize Rich for better output formatting and visualization
rich = Console()

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
    "row_factory": dict_row,
}


# Define a function to process chunks from the agent
def process_chunks(chunk):
    """
    Processes a chunk from the agent and displays information about tool calls or the agent's answer.
    """
    # Check if the chunk contains an agent's message
    if "agent" in chunk:
        # Iterate over the messages in the chunk
        for message in chunk["agent"]["messages"]:
            # Check if the message contains tool calls
            if "tool_calls" in message.additional_kwargs:
                # If the message contains tool calls, extract and display an informative message with tool call details

                # Extract all the tool calls
                tool_calls = message.additional_kwargs["tool_calls"]

                # Iterate over the tool calls
                for tool_call in tool_calls:
                    # Extract the tool name
                    tool_name = tool_call["function"]["name"]

                    # Extract the tool query
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    tool_query = tool_arguments.get("query", tool_arguments.get("city", "unknown"))

                    # Display an informative message with tool call details
                    rich.print(
                        f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] with the query [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]. Please wait for the agent's answer[deep_sky_blue1]...[/deep_sky_blue1]",
                        style="deep_sky_blue1",
                    )
            else:
                # If the message doesn't contain tool calls, extract and display the agent's answer

                # Extract the agent's answer
                agent_answer = message.content

                # Display the agent's answer
                rich.print(f"\nAgent:\n{agent_answer}", style="black on white")


# Define an async function to process checkpoints from the memory
async def process_checkpoints(checkpoints):
    """
    Asynchronously processes a list of checkpoints and displays relevant information.
    """
    rich.print("\n==========================================================\n")

    # Initialize an empty list to store the checkpoints
    checkpoints_list = []

    # Iterate over the checkpoints and add them to the list in an async way
    async for checkpoint_tuple in checkpoints:
        checkpoints_list.append(checkpoint_tuple)

    # Iterate over the list of checkpoints
    for idx, checkpoint_tuple in enumerate(checkpoints_list):
        # Extract key information about the checkpoint
        checkpoint = checkpoint_tuple.checkpoint
        messages = checkpoint["channel_values"].get("messages", [])

        # Display checkpoint information
        rich.print(f"[white]Checkpoint:[/white]")
        rich.print(f"[black]Timestamp: {checkpoint['ts']}[/black]")
        rich.print(f"[black]Checkpoint ID: {checkpoint['id']}[/black]")

        # Display checkpoint messages
        for message in messages:
            if isinstance(message, HumanMessage):
                rich.print(
                    f"[bright_magenta]User: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )
            elif isinstance(message, AIMessage):
                rich.print(
                    f"[bright_magenta]Agent: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )

        rich.print("")

    rich.print("==========================================================")


async def main():
    """
    Entry point of the application with conversational interface.
    """
    # Connect to the PostgreSQL database using an async connection pool
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        try:
            # Try to set up, but catch the error if tables already exist
            await checkpointer.setup()
        except psycopg.errors.DuplicateTable:
            rich.print("Tables already exist, continuing...")
        except psycopg.errors.DuplicatePreparedStatement:
            rich.print("Duplicate prepared statement encountered, continuing...")
        
        # Create a session ID for this conversation
        session_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": session_id},
            "name_prefix": f"session_{session_id}"
        }

        # Create a LangGraph agent
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
        
        rich.print("[bold green]Welcome to the Weather Assistant![/bold green]")
        rich.print("You can ask about weather in NYC or SF. Type 'quit' to exit.")
        
        # Loop until the user chooses to quit the chat
        while True:
            # Get the user's question and display it in the terminal
            user_question = input("\nUser:\n")

            # Check if the user wants to quit the chat
            if user_question.lower() in ["quit", "exit", "bye"]:
                rich.print(
                    "\nAgent:\nHave a nice day! :wave:\n", style="black on white"
                )
                break

            # Use the async stream method of the LangGraph agent to get the agent's answer
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=user_question)]},
                config,
            ):
                # Process the chunks from the agent
                process_chunks(chunk)

            # Use the async list method of the memory to list all checkpoints that match the configuration
            checkpoints = checkpointer.alist(config)
            # Process the checkpoints from the memory in an async way
            await process_checkpoints(checkpoints)


# Run the async main function
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
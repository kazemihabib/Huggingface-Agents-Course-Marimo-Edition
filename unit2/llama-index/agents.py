# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chromadb==0.6.3",
#     "datasets==3.4.0",
#     "huggingface-hub==0.29.3",
#     "llama-index==0.12.24",
#     "llama-index-callbacks-arize-phoenix==0.4.0",
#     "llama-index-embeddings-huggingface-api==0.3.0",
#     "llama-index-llms-huggingface-api==0.4.1",
#     "llama-index-vector-stores-chroma==0.4.1",
#     "marimo",
#     "python-dotenv==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Agents in LlamaIndex

        This notebook is part of the [Hugging Face Agents Course](https://www.hf.co/learn/agents-course), a free Course from beginner to expert, where you learn to build Agents.

        ![Agents course share](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png)

        ## Let's install the dependencies

        We will install the dependencies for this unit.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
    # !pip install llama-index datasets llama-index-callbacks-arize-phoenix llama-index-vector-stores-chroma llama-index-llms-huggingface-api -U -q
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And, let's log in to Hugging Face to use serverless Inference APIs.""")
    return


@app.cell
def _(os):
    from huggingface_hub import login

    login(os.environ["HF_TOKEN"])
    return (login,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Initialising agents

        Let's start by initialising an agent. We will use the basic `AgentWorkflow` class to create an agent.
        """
    )
    return


@app.cell
def _():
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream


    def _add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b


    def _subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b


    def _multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b


    def _divide(a: int, b: int) -> int:
        """Divide two numbers"""
        return a / b


    _llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[_subtract, _multiply, _divide, _add],
        llm=_llm,
        system_prompt="You are a math agent that can add, subtract, multiply, and divide numbers using provided tools.",
    )
    return (
        AgentStream,
        AgentWorkflow,
        HuggingFaceInferenceAPI,
        ToolCallResult,
        agent,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then, we can run the agent and get the response and reasoning behind the tool calls.""")
    return


@app.cell
async def _(AgentStream, ToolCallResult, agent):
    _handler = agent.run("What is (2 + 2) * 2?")
    async for _ev in _handler.stream_events():
        if isinstance(_ev, ToolCallResult):
            print("")
            print(
                "Called tool: ",
                _ev.tool_name,
                _ev.tool_kwargs,
                "=>",
                _ev.tool_output,
            )
        elif isinstance(_ev, AgentStream):  # showing the thought process
            print(_ev.delta, end="", flush=True)

    resp = await _handler
    resp
    return (resp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In a similar fashion, we can pass state and context to the agent.""")
    return


@app.cell
async def _(agent):
    from llama_index.core.workflow import Context

    _ctx = Context(agent)

    response = await agent.run("My name is Bob.", ctx=_ctx)
    response = await agent.run("What was my name again?", ctx=_ctx)
    response
    return Context, response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating RAG Agents with QueryEngineTools

        Let's now re-use the `QueryEngine` we defined in the [previous unit on tools](/tools.ipynb) and convert it into a `QueryEngineTool`. We will pass it to the `AgentWorkflow` class to create a RAG agent.
        """
    )
    return


@app.cell
def _(AgentWorkflow, HuggingFaceInferenceAPI):
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.huggingface_api import (
        HuggingFaceInferenceAPIEmbedding,
    )
    from llama_index.core.tools import QueryEngineTool
    from llama_index.vector_stores.chroma import ChromaVectorStore

    _db = chromadb.PersistentClient(path="./alfred_chroma_db")
    _chroma_collection = _db.get_or_create_collection("alfred")
    _vector_store = ChromaVectorStore(chroma_collection=_chroma_collection)
    _embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    _index = VectorStoreIndex.from_vector_store(
        vector_store=_vector_store, embed_model=_embed_model
    )
    _query_engine = _index.as_query_engine(llm=llm)
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=_query_engine,
        name="personas",
        description="descriptions for various types of personas",
        return_direct=False,
    )
    query_engine_agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_engine_tool],
        llm=llm,
        system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. ",
    )
    return (
        ChromaVectorStore,
        HuggingFaceInferenceAPIEmbedding,
        QueryEngineTool,
        VectorStoreIndex,
        chromadb,
        llm,
        query_engine_agent,
        query_engine_tool,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And, we can once more get the response and reasoning behind the tool calls.""")
    return


@app.cell
async def _(AgentStream, ToolCallResult, query_engine_agent):
    _handler = query_engine_agent.run(
        "Search the database for 'science fiction' and return some persona descriptions."
    )
    async for _ev in _handler.stream_events():
        if isinstance(_ev, ToolCallResult):
            print("")
            print(
                "Called tool: ",
                _ev.tool_name,
                _ev.tool_kwargs,
                "=>",
                _ev.tool_output,
            )
        elif isinstance(_ev, AgentStream):
            print(_ev.delta, end="", flush=True)
    resp_1 = await _handler
    resp_1
    return (resp_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating multi-agent systems

        We can also create multi-agent systems by passing multiple agents to the `AgentWorkflow` class.
        """
    )
    return


@app.cell
def _(AgentWorkflow, llm, query_engine_tool):
    from llama_index.core.agent.workflow import ReActAgent


    def _add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b


    def _subtract(a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b


    _calculator_agent = ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant. Use your tools for any math operation.",
        tools=[_add, _subtract],
        llm=llm,
    )
    _query_agent = ReActAgent(
        name="info_lookup",
        description="Looks up information about XYZ",
        system_prompt="Use your tool to query a RAG system to answer information about XYZ",
        tools=[query_engine_tool],
        llm=llm,
    )
    _agent = AgentWorkflow(
        agents=[_calculator_agent, _query_agent], root_agent="calculator"
    )
    handler_2 = _agent.run(user_msg="Can you add 5 and 3?")
    return ReActAgent, handler_2


@app.cell
async def _(AgentStream, ToolCallResult, handler_2):
    async for _ev in handler_2.stream_events():
        if isinstance(_ev, ToolCallResult):
            print('')
            print('Called tool: ', _ev.tool_name, _ev.tool_kwargs, '=>', _ev.tool_output)
        elif isinstance(_ev, AgentStream):
            print(_ev.delta, end='', flush=True)
    resp_2 = await handler_2
    resp_2
    return (resp_2,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    load_dotenv(".env")
    return load_dotenv, os


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

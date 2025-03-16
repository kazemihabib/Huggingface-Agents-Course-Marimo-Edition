# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chromadb==0.6.3",
#     "datasets==3.4.0",
#     "huggingface-hub==0.29.3",
#     "llama-index==0.12.24",
#     "llama-index-callbacks-arize-phoenix==0.4.0",
#     "llama-index-embeddings-huggingface==0.5.2",
#     "llama-index-llms-huggingface-api==0.4.1",
#     "llama-index-tools-google==0.3.0",
#     "llama-index-vector-stores-chroma==0.4.1",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Tools in LlamaIndex


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
        ## Creating a FunctionTool

        Let's create a basic `FunctionTool` and call it.
        """
    )
    return


@app.cell
def _():
    from llama_index.core.tools import FunctionTool


    def _get_weather(location: str) -> str:
        """Useful for getting the weather for a given location."""
        print(f"Getting weather for {location}")
        return f"The weather in {location} is sunny"


    _tool = FunctionTool.from_defaults(
        _get_weather,
        name="my_weather_tool",
        description="Useful for getting the weather for a given location.",
    )
    _tool.call("New York")
    return (FunctionTool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating a QueryEngineTool

        Let's now re-use the `QueryEngine` we defined in the [previous unit on tools](/tools.ipynb) and convert it into a `QueryEngineTool`.
        """
    )
    return


@app.cell
async def _():
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
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
    _llm = HuggingFaceInferenceAPI(model_name="meta-llama/Llama-3.2-3B-Instruct")
    _index = VectorStoreIndex.from_vector_store(
        vector_store=_vector_store, embed_model=_embed_model
    )
    _query_engine = _index.as_query_engine(llm=_llm)
    _tool = QueryEngineTool.from_defaults(
        query_engine=_query_engine,
        name="some useful name",
        description="some useful description",
    )
    await _tool.acall(
        "Responds about research on the impact of AI on the future of work and society?"
    )
    return (
        ChromaVectorStore,
        HuggingFaceInferenceAPI,
        HuggingFaceInferenceAPIEmbedding,
        QueryEngineTool,
        VectorStoreIndex,
        chromadb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating Toolspecs

        Let's create a `ToolSpec` from the `GmailToolSpec` from the LlamaHub and convert it to a list of tools.
        """
    )
    return


@app.cell
def _():
    from llama_index.tools.google import GmailToolSpec

    _tool_spec = GmailToolSpec()
    tool_spec_list = _tool_spec.to_tool_list()
    tool_spec_list
    return GmailToolSpec, tool_spec_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To get a more detailed view of the tools, we can take a look at the `metadata` of each tool.""")
    return


@app.cell
def _(tool_spec_list):
    [(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    return (os,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==3.4.0",
#     "huggingface-hub==0.29.3",
#     "llama-index==0.12.24",
#     "llama-index-callbacks-arize-phoenix==0.4.0",
#     "llama-index-embeddings-huggingface-api",
#     "llama-index-llms-huggingface-api==0.4.1",
#     "llama-index-vector-stores-chroma==0.4.1",
#     "llama-index-embeddings-huggingface==0.5.2",
#     "marimo",
#     "python-dotenv==1.0.1",
#     "chromadb==0.6.3",
#     "nest-asyncio==1.6.0",
#     "arize-phoenix==8.13.1",
# ]
# ///

import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Components in LlamaIndex

        This notebook is part of the [Hugging Face Agents Course](https://www.hf.co/learn/agents-course), a free Course from beginner to expert, where you learn to build Agents.

        ![Agents course share](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png)

        Alfred is hosting a party and needs to be able to find relevant information on personas that will be attending the party. Therefore, we will use a `QueryEngine` to index and search through a database of personas.

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
        ## Create a `QueryEngine` for retrieval augmented generation

        ### Setting up the persona database

        We will be using personas from the [dvilasuero/finepersonas-v0.1-tiny dataset](https://huggingface.co/datasets/dvilasuero/finepersonas-v0.1-tiny). This dataset contains 5K personas that will be attending the party!

        Let's load the dataset and store it as files in the `data` directory
        """
    )
    return


@app.cell
def _():
    from datasets import load_dataset
    from pathlib import Path

    _dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")

    Path("data").mkdir(parents=True, exist_ok=True)
    for _i, _persona in enumerate(_dataset):
        with open(Path("data") / f"persona_{_i}.txt", "w") as _f:
            _f.write(_persona["persona"])
    return Path, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Awesome, now we have a local directory with all the personas that will be attending the party, we can load and index!

        ### Loading and embedding persona documents

        We will use the `SimpleDirectoryReader` to load the persona descriptions from the `data` directory. This will return a list of `Document` objects.
        """
    )
    return


@app.cell
def _():
    from llama_index.core import SimpleDirectoryReader

    _reader = SimpleDirectoryReader(input_dir="data")
    documents = _reader.load_data()
    len(documents)
    return SimpleDirectoryReader, documents


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we have a list of `Document` objects, we can use the `IngestionPipeline` to create nodes from the documents and prepare them for the `QueryEngine`. We will use the `SentenceSplitter` to split the documents into smaller chunks and the `HuggingFaceInferenceAPIEmbedding` to embed the chunks.""")
    return


@app.cell
async def _(documents):
    from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.ingestion import IngestionPipeline

    # create the pipeline with transformations
    _pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ]
    )

    # run the pipeline sync or async
    _nodes = await _pipeline.arun(documents=documents[:10])
    _nodes
    return (
        HuggingFaceInferenceAPIEmbedding,
        IngestionPipeline,
        SentenceSplitter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As, you can see, we have created a list of `Node` objects, which are just chunks of text from the original documents. Let's explore how we can add these nodes to a vector store.

        ### Storing and indexing documents

        Since we are using an ingestion pipeline, we can directly attach a vector store to the pipeline to populate it.
        In this case, we will use `Chroma` to store our documents.
        Let's run the pipeline again with the vector store attached. 
        The `IngestionPipeline` caches the operations so this should be fast!
        """
    )
    return


@app.cell
async def _(
    HuggingFaceInferenceAPIEmbedding,
    IngestionPipeline,
    SentenceSplitter,
    documents,
):
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    _db = chromadb.PersistentClient(path="./alfred_chroma_db")
    _chroma_collection = _db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=_chroma_collection)
    _pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )
    _nodes = await _pipeline.arun(documents=documents[:10])
    len(_nodes)
    return ChromaVectorStore, chromadb, vector_store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can create a `VectorStoreIndex` from the vector store and use it to query the documents by passing the vector store and embedding model to the `from_vector_store()` method.""")
    return


@app.cell
def _(HuggingFaceInferenceAPIEmbedding, vector_store):
    from llama_index.core import VectorStoreIndex

    _embed_model = HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=_embed_model
    )
    return VectorStoreIndex, index


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We don't need to worry about persisting the index to disk, as it is automatically saved within the `ChromaVectorStore` object and the passed directory path.

        ### Querying the index

        Now that we have our index, we can use it to query the documents.
        Let's create a `QueryEngine` from the index and use it to query the documents using a specific response mode.
        """
    )
    return


@app.cell
def _(index):
    # from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.llms.ollama import Ollama
    import nest_asyncio


    nest_asyncio.apply()  # This is needed to run the query engine
    llm = Ollama(model="qwen2.5-coder:7b", request_timeout=60.0)
    # I am going to use local LLM, you can uncomment the next line to use huggingfaceInferenceAPI
    # _llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    response = query_engine.query(
        "Respond using a persona that describes author and travel experiences?"
    )

    response
    return Ollama, llm, nest_asyncio, query_engine, response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Evaluation and observability

        LlamaIndex provides **built-in evaluation tools to assess response quality.**
        These evaluators leverage LLMs to analyze responses across different dimensions.
        We can now check if the query is faithful to the original persona.
        """
    )
    return


@app.cell
def _(llm, response):
    from llama_index.core.evaluation import FaithfulnessEvaluator

    # query index
    _evaluator = FaithfulnessEvaluator(llm=llm)
    _eval_result = _evaluator.evaluate_response(response=response)
    _eval_result.passing
    return (FaithfulnessEvaluator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If one of these LLM based evaluators does not give enough context, we can check the response using the Arize Phoenix tool, after creating an account at [LlamaTrace](https://llamatrace.com/login) and generating an API key.""")
    return


@app.cell
def _(os):
    import llama_index

    PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    llama_index.core.set_global_handler(
        "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
    )
    return PHOENIX_API_KEY, llama_index


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, we can query the index and see the response in the Arize Phoenix tool.""")
    return


@app.cell
def _(query_engine):
    response_1 = query_engine.query('What is the name of the someone that is interested in AI and techhnology?')
    response_1
    return (response_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can then go to the [LlamaTrace](https://llamatrace.com/login) and explore the process and response.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![arize-phoenix](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/arize.png)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from dotenv import load_dotenv

    load_dotenv(".env")  # take environment variables from .env.
    return load_dotenv, os


if __name__ == "__main__":
    app.run()

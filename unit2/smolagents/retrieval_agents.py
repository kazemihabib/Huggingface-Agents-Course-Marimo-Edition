# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.3",
#     "langchain==0.3.20",
#     "langchain-community==0.3.19",
#     "marimo",
#     "rank-bm25==0.2.2",
#     "smolagents==1.10.0",
# ]
# ///

import marimo

__generated_with = "0.11.19"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Building Agentic RAG Systems

        This notebook is part of the [Hugging Face Agents Course](https://www.hf.co/learn/agents-course), a free Course from beginner to expert, where you learn to build Agents.

        ![Agents course share](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Let's install the dependencies and login to our HF account to access the Inference API

        If you haven't installed `smolagents` yet, you can do so by running the following command:the dependencies and login to our HF account to access the Inference API
        """
    )
    return


@app.cell
def _(mo):
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
    # !pip install smolagents
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's also login to the Hugging Face Hub to have access to the Inference API.""")
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
        ## Basic Retrieval with DuckDuckGo

        Let's build a simple agent that can search the web using DuckDuckGo. This agent will retrieve information and synthesize responses to answer queries. With Agentic RAG, Alfred's agent can:

        * Search for latest superhero party trends
        * Refine results to include luxury elements
        * Synthesize information into a complete plan

        Here's how Alfred's agent can achieve this:
        """
    )
    return


@app.cell
def _(CodeAgent, DuckDuckGoSearchTool, HfApiModel, mo):
    # Initialize the search tool
    _search_tool = DuckDuckGoSearchTool()

    # Initialize the model
    _model = HfApiModel()

    _agent = CodeAgent(
        model = _model,
        tools=[_search_tool]
    )

    # Example usage
    _response = _agent.run(
        "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
    )

    if isinstance(_response, dict):
        mo.output.replace(_response)
    else:
        mo.md(_response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The agent follows this process:

        1. **Analyzes the Request:** Alfred’s agent identifies the key elements of the query—luxury superhero-themed party planning, with focus on decor, entertainment, and catering.
        2. **Performs Retrieval:**  The agent leverages DuckDuckGo to search for the most relevant and up-to-date information, ensuring it aligns with Alfred’s refined preferences for a luxurious event.
        3. **Synthesizes Information:** After gathering the results, the agent processes them into a cohesive, actionable plan for Alfred, covering all aspects of the party.
        4. **Stores for Future Reference:** The agent stores the retrieved information for easy access when planning future events, optimizing efficiency in subsequent tasks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Custom Knowledge Base Tool

        For specialized tasks, a custom knowledge base can be invaluable. Let's create a tool that queries a vector database of technical documentation or specialized knowledge. Using semantic search, the agent can find the most relevant information for Alfred's needs.

        This approach combines predefined knowledge with semantic search to provide context-aware solutions for event planning. With specialized knowledge access, Alfred can perfect every detail of the party.

        Install the dependecies first and run!
        """
    )
    return


@app.cell
def _(mo):
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
    # !pip install langchain-community rank_bm25
    return


@app.cell
def _(CodeAgent, HfApiModel, mo):
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from smolagents import Tool
    from langchain_community.retrievers import BM25Retriever


    class _PartyPlanningRetrieverTool(Tool):
        name = "party_planning_retriever"
        description = "Uses semantic search to retrieve relevant party planning ideas for Alfred’s superhero-themed party at Wayne Manor."
        inputs = {
            "query": {
                "type": "string",
                "description": "The query to perform. This should be a query related to party planning or superhero themes.",
            }
        }
        output_type = "string"

        def __init__(self, docs, **kwargs):
            super().__init__(**kwargs)
            self.retriever = BM25Retriever.from_documents(docs, k=5)

        def forward(self, query: str) -> str:
            assert isinstance(query, str), "Your search query must be a string"
            docs = self.retriever.invoke(query)
            return "\nRetrieved ideas:\n" + "".join(
                [
                    f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )


    party_ideas = [
        {
            "text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.",
            "source": "Party Ideas 1",
        },
        {
            "text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.",
            "source": "Entertainment Ideas",
        },
        {
            "text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'",
            "source": "Catering Ideas",
        },
        {
            "text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.",
            "source": "Decoration Ideas",
        },
        {
            "text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.",
            "source": "Entertainment Ideas",
        },
    ]
    _source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in party_ideas
    ]
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    _docs_processed = _text_splitter.split_documents(_source_docs)
    print("Rashid:", _docs_processed[0])
    print("Rashid:", _docs_processed[1])
    _party_planning_retriever = _PartyPlanningRetrieverTool(_docs_processed)
    _agent = CodeAgent(tools=[_party_planning_retriever], model=HfApiModel())
    _response = _agent.run(
        "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
    )

    mo.md(_response)
    return (
        BM25Retriever,
        Document,
        RecursiveCharacterTextSplitter,
        Tool,
        party_ideas,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This enhanced agent can:
        1. First check the documentation for relevant information
        2. Combine insights from the knowledge base
        3. Maintain conversation context through memory
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
    return CodeAgent, DuckDuckGoSearchTool, HfApiModel, os


if __name__ == "__main__":
    app.run()

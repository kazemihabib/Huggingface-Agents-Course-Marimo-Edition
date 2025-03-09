# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.2",
#     "marimo",
#     "smolagents==1.10.0",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full", app_title="Integrating Agents With Tools")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Integrating Agents With Tools

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

        If you haven't installed `smolagents` yet, you can do so by running the following command:
        """
    )
    return


@app.cell
def _(mo):
    # !pip install smolagents -U
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
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
        ## Selecting a Playlist for the Party Using `smolagents` and a `ToolCallingAgent`

        Let's revisit the previous example where Alfred started party preparations, but this time we'll use a `ToolCallingAgent` to highlight the difference. We'll build an agent that can search the web using DuckDuckGo, just like in our Code Agent example. The only difference is the agent type - the framework handles everything else:
        """
    )
    return


@app.cell
def _():
    # model_id='Qwen/Qwen2.5-Coder-32B-Instruct'# it is possible that this model may be overloaded

    # If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
    model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 
    return (model_id,)


@app.cell
def _(mo, model_id):
    from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel

    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel(model_id))

    mo.md(agent.run("Search for the best music recommendations for a party at the Wayne's mansion."))
    return DuckDuckGoSearchTool, HfApiModel, ToolCallingAgent, agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When you examine the agent's trace, instead of seeing `Executing parsed code:`, you'll see something like:

        ```text
        ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
        │ Calling tool: 'web_search' with arguments: {'query': "best music recommendations for a party at Wayne's         │
        │ mansion"}                                                                                                       │
        ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
        ```  

        The agent generates a structured tool call that the system processes to produce the output, rather than directly executing code like a `CodeAgent`.

        Now that we understand both agent types, we can choose the right one for our needs. Let's continue exploring `smolagents` to make Alfred's party a success! 🎉
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    return mo, os


if __name__ == "__main__":
    app.run()

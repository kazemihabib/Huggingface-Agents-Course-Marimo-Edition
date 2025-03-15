# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.3",
#     "marimo",
#     "pillow==11.1.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
#     "smolagents[openai]==1.11.0",
# ]
# ///

import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Vision Agents with smolagents


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
        ## Providing Images at the Start of the Agent's Execution

        In this approach, images are passed to the agent at the start and stored as `task_images` alongside the task prompt. The agent then processes these images throughout its execution.  

        Consider the case where Alfred wants to verify the identities of the superheroes attending the party. He already has a dataset of images from previous parties with the names of the guests. Given a new visitor's image, the agent can compare it with the existing dataset and make a decision about letting them in.  

        In this case, a guest is trying to enter, and Alfred suspects that this visitor might be The Joker impersonating Wonder Woman. Alfred needs to verify their identity to prevent anyone unwanted from entering.  

        Let’s build the example. First, the images are loaded. In this case, we use images from Wikipedia to keep the example minimal, but image the possible use-case!
        """
    )
    return


@app.cell
def _():
    from PIL import Image
    import requests
    from io import BytesIO

    _image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
        "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
    ]

    images = []
    for _url in _image_urls:
        _response = requests.get(_url)
        _image = Image.open(BytesIO(_response.content)).convert("RGB")
        images.append(_image)
    return BytesIO, Image, images, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have the images, the agent will tell us wether the guests is actually a superhero (Wonder Woman) or a villian (The Joker).""")
    return


@app.cell
def _(mo):
    mo.md("Please set your OpenAI API key in the `OPENAI_API_KEY` environment variable")
    return


@app.cell
def _(images):
    from smolagents import CodeAgent, OpenAIServerModel
    _model = OpenAIServerModel(model_id='gpt-4o')
    _agent = CodeAgent(tools=[], model=_model, max_steps=20, verbosity_level=2)
    response = _agent.run('\n    Describe the costume and makeup that the comic character in these photos is wearing and return the description.\n    Tell me if the guest is The Joker or Wonder Woman.\n    ', images=images)
    return CodeAgent, OpenAIServerModel, response


@app.cell
def _(response):
    response
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this case, the output reveals that the person is impersonating someone else, so we can prevent The Joker from entering the party!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Providing Images with Dynamic Retrieval

        This examples is provided as a `.py` file since it needs to be run locally since it'll browse the web. Go to the [Hugging Face Agents Course](https://www.hf.co/learn/agents-course) for more details.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path=".env", verbose=True)
    return load_dotenv, mo, os


if __name__ == "__main__":
    app.run()

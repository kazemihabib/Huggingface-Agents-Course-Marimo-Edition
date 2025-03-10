# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.2",
#     "marimo",
#     "pillow==11.1.0",
#     "smolagents==1.10.0",
#     "gradio_client==1.7.2",
#     "matplotlib==3.10.1",
#     "langchain-community==0.3.19",
#     "google-search-results==2.4.2",
#     "langchain==0.3.20",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full", app_title="Building Agents That Use Code")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Building Agents That Use Code

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
    mo.md(
        "We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv."
    )
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
        ## The `@tool` Decorator  

        ### Generating a tool that retrieves the highest-rated catering

        Let's imagine that Alfred has already decided on the menu for the party, but now he needs help preparing food for such a large number of guests. To do so, he would like to hire a catering service and needs to identify the highest-rated options available. Alfred can leverage a tool to search for the best catering services in his area.

        Below is an example of how Alfred can use the `@tool` decorator to make this happen:
        """
    )
    return


@app.cell
def _():
    # model_id='Qwen/Qwen2.5-Coder-32B-Instruct'# it is possible that this model may be overloaded

    # If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
    model_id = "https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud"
    return (model_id,)


@app.cell
def _(CodeAgent, HfApiModel, mo, model_id, tool):
    # Let's pretend we have a function that fetches the highest-rated catering services.
    @tool
    def _catering_service_tool(query: str) -> str:
        """
        This tool returns the highest-rated catering service in Gotham City.

        Args:
            query: A search term for finding catering services.
        """
        # Example list of catering services and their ratings
        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }

        # Find the highest rated catering service (simulating search query filtering)
        best_service = max(services, key=services.get)

        return best_service


    _agent = CodeAgent(tools=[_catering_service_tool], model=HfApiModel(model_id))

    # Run the agent to find the best catering service
    _result = _agent.run(
        "Can you give me the name of the highest-rated catering service in Gotham City?"
    )

    mo.md(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Defining a Tool as a Python Class  

        ### Generating a tool to generate ideas about the superhero-themed party

        Alfred's party at the mansion is a **superhero-themed event**, but he needs some creative ideas to make it truly special. As a fantastic host, he wants to surprise the guests with a unique theme.

        To do this, he can use an agent that generates superhero-themed party ideas based on a given category. This way, Alfred can find the perfect party theme to wow his guests.
        """
    )
    return


@app.cell
def _(CodeAgent, HfApiModel, Tool, mo, model_id):
    class SuperheroPartyThemeTool(Tool):
        name = "superhero_party_theme_generator"
        description = "\n    This tool suggests creative superhero-themed party ideas based on a category.\n    It returns a unique party theme idea."
        inputs = {
            "category": {
                "type": "string",
                "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
            }
        }
        output_type = "string"

        def forward(self, category: str):
            themes = {
                "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
                "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
                "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets.",
            }
            return themes.get(
                category.lower(),
                "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.",
            )


    party_theme_tool = SuperheroPartyThemeTool()
    _agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel(model_id))
    _result = _agent.run(
        "What would be a good superhero party idea for a 'villain masquerade' theme?"
    )
    mo.md(_result)
    return SuperheroPartyThemeTool, party_theme_tool


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sharing a Tool to the Hub

        Sharing your custom tool with the community is easy! Simply upload it to your Hugging Face account using the `push_to_hub()` method.

        For instance, Alfred can share his `catering_service_tool` to help others find the best catering services in Gotham. Here's how to do it:
        """
    )
    return


@app.cell
def _(mo):
    ### TODO(Find a way to make push_to_hub work in marimo)
    mo.md(
        """ Unfortunately, The push_to_hub function fails in Marimo notebooks because: It relies on IPython to extract code from notebook cells and Marimo does not use the IPython environment
        """
    )

    # party_theme_tool.push_to_hub("{your_username}/catering_service_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Importing a Tool from the Hub

        You can easily import tools created by other users using the `load_tool()` function. For example, Alfred might want to generate a promotional image for the party using AI. Instead of building a tool from scratch, he can leverage a predefined one from the community:
        """
    )
    return


@app.cell
def _(CodeAgent, HfApiModel, load_tool):
    _image_generation_tool = load_tool(
        "m-ric/text-to-image", trust_remote_code=True
    )
    _agent = CodeAgent(tools=[_image_generation_tool], model=HfApiModel())
    _agent.run(
        "Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Importing a Hugging Face Space as a Tool

        You can also import a HF Space as a tool using `Tool.from_space()`. This opens up possibilities for integrating with thousands of spaces from the community for tasks from image generation to data analysis.

        The tool will connect with the spaces Gradio backend using the `gradio_client`, so make sure to install it via `pip` if you don't have it already. For the party, Alfred can also use a HF Space directly for the generation of the previous annoucement AI-generated image. Let's build it!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        "We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv."
    )
    # !pip install gradio_client
    return


@app.cell
def _(CodeAgent, HfApiModel, Tool):
    _image_generation_tool = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="Generate an image from a prompt",
    )
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
    _agent = CodeAgent(tools=[_image_generation_tool], model=model)
    _agent.run(
        "Improve this prompt, then generate an image of it.",
        additional_args={
            "user_prompt": "A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala"
        },
    )
    return (model,)


@app.cell
def _():
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt

    _image_path = "/private/var/folders/q1/7g9hgvc17m95t7yrvcclc75h0000gn/T/gradio/5161a771b26474fa6f608f420ed6a9d4d034f9155cdd705615610841f31cc301/image.webp"

    _img = PILImage.open(_image_path)
    _img
    return PILImage, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Importing a LangChain Tool

        These tools need a [SerpApi API Key](https://serpapi.com/).

        You can easily load LangChain tools using the `Tool.from_langchain()` method. Alfred, ever the perfectionist, is preparing for a spectacular superhero night at Wayne Manor while the Waynes are away. To make sure every detail exceeds expectations, he taps into LangChain tools to find top-tier entertainment ideas.

        By using `Tool.from_langchain()`, Alfred effortlessly adds advanced search functionalities to his smolagent, enabling him to discover exclusive party ideas and services with just a few commands.

        Here's how he does it:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        "We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv."
    )
    # !pip install langchain-community google-search-results
    return


@app.cell
def _(mo):
    mo.md("Add the SerpApi API Key to the environment variables: SERPAPI_API_KEY")
    mo.md("""
    ### 🔑 API Key Configuration

    To use the search functionality, you need to add your SerpAPI key to your environment:

    ```bash
    export SERPAPI_API_KEY="your_api_key_here"
    ```
    """)
    return


@app.cell
def _(CodeAgent, Tool, model):
    from langchain.agents import load_tools

    _search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
    _agent = CodeAgent(tools=[_search_tool], model=model)
    _agent.run(
        "Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences."
    )
    return (load_tools,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With this setup, Alfred can quickly discover luxurious entertainment options, ensuring Gotham's elite guests have an unforgettable experience. This tool helps him curate the perfect superhero-themed event for Wayne Manor! 🎉""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from smolagents import CodeAgent, HfApiModel, Tool, tool, load_tool
    return CodeAgent, HfApiModel, Tool, load_tool, os, tool


if __name__ == "__main__":
    app.run()

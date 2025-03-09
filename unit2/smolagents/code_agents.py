# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.2",
#     "marimo",
#     "numpy==2.2.3",
#     "openinference-instrumentation-smolagents==0.1.6",
#     "opentelemetry-exporter-otlp==1.30.0",
#     "opentelemetry-sdk==1.30.0",
#     "smolagents==1.10.0",
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

        Alfred is planning a party at the Wayne family mansion and needs your help to ensure everything goes smoothly. To assist him, we'll apply what we've learned about how a multi-step `CodeAgent` operates.
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
    # !pip install smolagents -
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's also login to the Hugging Face Hub to have access to the Inference API.""")
    return


@app.cell
def _(os):
    #TODO("Move to setup cell once marimo finished implementing it.")
    from huggingface_hub import login

    login(os.environ["HF_TOKEN"])
    return (login,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Selecting a Playlist for the Party Using `smolagents`

        An important part of a successful party is the music. Alfred needs some help selecting the playlist. Luckily, `smolagents` has got us covered! We can build an agent capable of searching the web using DuckDuckGo. To give the agent access to this tool, we include it in the tool list when creating the agent.

        For the model, we'll rely on `HfApiModel`, which provides access to Hugging Face's [Inference API](https://huggingface.co/docs/api-inference/index). The default model is `"Qwen/Qwen2.5-Coder-32B-Instruct"`, which is performant and available for fast inference, but you can select any compatible model from the Hub.

        Running an agent is quite straightforward:
        """
    )
    return


@app.cell
def _(CodeAgent, DuckDuckGoSearchTool, HfApiModel):
    # model_id='Qwen/Qwen2.5-Coder-32B-Instruct'# it is possible that this model may be overloaded

    # If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
    model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 


    _agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel(model_id))

    _agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
    return (model_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When you run this example, the output will **display a trace of the workflow steps being executed**. It will also print the corresponding Python code with the message:

        ```python
         ─ Executing parsed code: ────────────────────────────────────────────────────────────────────────────────────────
          results = web_search(query="best music for a Batman party")                                                      
          print(results)                                                                                                   
         ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        ```

        After a few steps, you'll see the generated playlist that Alfred can use for the party! 🎵
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using a Custom Tool to Prepare the Menu

        Now that we have selected a playlist, we need to organize the menu for the guests. Again, Alfred can take advantage of `smolagents` to do so. Here, we use the `@tool` decorator to define a custom function that acts as a tool. We'll cover tool creation in more detail later, so for now, we can simply run the code.

        As you can see in the example below, we will create a tool using `@tool` decorator and include it in the `tools` list.
        """
    )
    return


@app.cell
def _(CodeAgent, HfApiModel, model_id, tool):
    @tool
    def _suggest_menu(occasion: str) -> str:
        """
        Suggests a menu based on the occasion.
        Args:
            occasion: The type of occasion for the party.
        """
        if occasion == 'casual':
            return 'Pizza, snacks, and drinks.'
        elif occasion == 'formal':
            return '3-course dinner with wine and dessert.'
        elif occasion == 'superhero':
            return 'Buffet with high-energy and healthy food.'
        else:
            return 'Custom menu for the butler.'
    _agent = CodeAgent(tools=[_suggest_menu], model=HfApiModel(model_id))
    _agent.run('Prepare a formal menu for the party.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The agent will run for a few steps until finding the answer.

        The menu is ready! 🥗
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using Python Imports Inside the Agent

        We have the playlist and menu ready, but we need to check one more crucial detail: preparation time!

        Alfred needs to calculate when everything would be ready if he started preparing now, in case they need assistance from other superheroes.

        `smolagents` specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.

        **Code execution has strict security measures** - imports outside a predefined safe list are blocked by default. However, you can authorize additional imports by passing them as strings in `additional_authorized_imports`.
        For more details on secure code execution, see the official [guide](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution).

        When creating the agent, we ill use `additional_authorized_imports` to allow for importing the `datetime` module.
        """
    )
    return


@app.cell
def _(CodeAgent, HfApiModel, mo, model_id):
    import numpy as np
    import time
    import datetime
    _agent = CodeAgent(tools=[], model=HfApiModel(model_id), additional_authorized_imports=['datetime'])
    mo.md(_agent.run('\n    Alfred needs to prepare for the party. Here are the tasks:\n    1. Prepare the drinks - 30 minutes\n    2. Decorate the mansion - 60 minutes\n    3. Set up the menu - 45 minutes\n    3. Prepare the music and playlist - 45 minutes\n\n    If we start right now, at what time will the party be ready?\n    '))
    return datetime, np, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These examples are just the beginning of what you can do with code agents, and we're already starting to see their utility for preparing the party.
        You can learn more about how to build code agents in the [smolagents documentation](https://huggingface.co/docs/smolagents).

        `smolagents` specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sharing Our Custom Party Preparator Agent to the Hub

        Wouldn't it be **amazing to share our very own Alfred agent with the community**? By doing so, anyone can easily download and use the agent directly from the Hub, bringing the ultimate party planner of Gotham to their fingertips! Let's make it happen! 🎉

        The `smolagents` library makes this possible by allowing you to share a complete agent with the community and download others for immediate use. It's as simple as the following:
        """
    )
    return


@app.cell
def _(
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    Tool,
    VisitWebpageTool,
    model_id,
    tool,
):
    @tool
    def _suggest_menu(occasion: str) -> str:
        """
        Suggests a menu based on the occasion.
        Args:
            occasion: The type of occasion for the party.
        """
        if occasion == 'casual':
            return 'Pizza, snacks, and drinks.'
        elif occasion == 'formal':
            return '3-course dinner with wine and dessert.'
        elif occasion == 'superhero':
            return 'Buffet with high-energy and healthy food.'
        else:
            return 'Custom menu for the butler.'

    @tool
    def _catering_service_tool(query: str) -> str:
        """
        This tool returns the highest-rated catering service in Gotham City.

        Args:
            query: A search term for finding catering services.
        """
        services = {'Gotham Catering Co.': 4.9, 'Wayne Manor Catering': 4.8, 'Gotham City Events': 4.7}
        best_service = max(services, key=services.get)
        return best_service

    class _SuperheroPartyThemeTool(Tool):
        name = 'superhero_party_theme_generator'
        description = '\n    This tool suggests creative superhero-themed party ideas based on a category.\n    It returns a unique party theme idea.'
        inputs = {'category': {'type': 'string', 'description': "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham')."}}
        output_type = 'string'

        def forward(self, category: str):
            themes = {'classic heroes': "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.", 'villain masquerade': "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.", 'futuristic Gotham': 'Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets.'}
            return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), _suggest_menu, _catering_service_tool, _SuperheroPartyThemeTool()], model=HfApiModel(model_id), max_steps=10, verbosity_level=2)
    agent.run("Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")
    return (agent,)


@app.cell
def _(mo):
    mo.md("""
            ### TODO(Find a way to push agent to hub in marino)
            Unfortunately, `push_to_hub` use `IPython` to get the code inside the cells.
            As marino not uses IPython it will throw error
    """)
    # agent.push_to_hub('habibkazemi2/AlfredAgent2')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To download the agent again, use the code below:""")
    return


@app.cell
def _(CodeAgent, HfApiModel):
    _agent = CodeAgent(tools=[], model=HfApiModel())
    alfred_agent = _agent.from_hub('habibkazemi2/AlfredAgent', trust_remote_code=True)
    return (alfred_agent,)


@app.cell
def _(alfred_agent):
    alfred_agent.run("Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What's also exciting is that shared agents are directly available as Hugging Face Spaces, allowing you to interact with them in real-time. You can explore other agents [here](https://huggingface.co/spaces/davidberenstein1957/smolagents-and-tools).

        For example, the _AlfredAgent_ is available [here](https://huggingface.co/spaces/sergiopaniego/AlfredAgent).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inspecting Our Party Preparator Agent with OpenTelemetry and Langfuse 📡

        Full trace can be found [here](https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10%3A28%3A36.929Z).

        As Alfred fine-tunes the Party Preparator Agent, he's growing weary of debugging its runs. Agents, by nature, are unpredictable and difficult to inspect. But since he aims to build the ultimate Party Preparator Agent and deploy it in production, he needs robust traceability for future monitoring and analysis.  

        Once again, `smolagents` comes to the rescue! It embraces the [OpenTelemetry](https://opentelemetry.io/) standard for instrumenting agent runs, allowing seamless inspection and logging. With the help of [Langfuse](https://langfuse.com/) and the `SmolagentsInstrumentor`, Alfred can easily track and analyze his agent’s behavior.  

        Setting it up is straightforward!  

        First, we need to install the necessary dependencies:
        """
    )
    return


@app.cell
def _(mo):
    mo.md("We don't need this cell anymore thanks to https://peps.python.org/pep-0723/ and uv.")
    # !pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, Alfred has already created an account on Langfuse and has his API keys ready. If you haven’t done so yet, you can sign up for Langfuse Cloud [here](https://cloud.langfuse.com/) or explore [alternatives](https://huggingface.co/docs/smolagents/tutorials/inspect_runs).  

        Once you have your API keys, they need to be properly configured as follows:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, Alfred is ready to initialize the `SmolagentsInstrumentor` and start tracking his agent's performance.""")
    return


@app.cell
def _(mo):
    mo.md("""Enable the following cell by right-clicking on it and enabling `Reactive Execution`.""")
    return


@app.cell(disabled=True)
def _(os):
    import base64

    LANGFUSE_PUBLIC_KEY=os.environ["LANGFUSE_PUBLIC_KEY"]
    LANGFUSE_SECRET_KEY=os.environ["LANGFUSE_SECRET_KEY"]
    LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
    # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
    from opentelemetry.sdk.trace import TracerProvider

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
    return (
        LANGFUSE_AUTH,
        LANGFUSE_PUBLIC_KEY,
        LANGFUSE_SECRET_KEY,
        OTLPSpanExporter,
        SimpleSpanProcessor,
        SmolagentsInstrumentor,
        TracerProvider,
        base64,
        trace_provider,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Alfred is now connected 🔌! The runs from `smolagents` are being logged in Langfuse, giving him full visibility into the agent's behavior. With this setup, he's ready to revisit previous runs and refine his Party Preparator Agent even further.""")
    return


@app.cell
def _(CodeAgent, HfApiModel, mo, model_id, trace_provider):
    assert trace_provider is not None
    _agent = CodeAgent(tools=[], model=HfApiModel(model_id))
    _alfred_agent = _agent.from_hub('habibkazemi2/AlfredAgent', trust_remote_code=True)
    mo.md(_alfred_agent.run("Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Alfred can now access this logs [here](https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10%3A28%3A36.929Z) to review and analyze them.  

        Meanwhile, the [suggested playlist](https://open.spotify.com/playlist/0gZMMHjuxMrrybQ7wTMTpw) sets the perfect vibe for the party preparations. Cool, right? 🎶
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
    return (os,)


@app.cell
def _():
    from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, VisitWebpageTool, FinalAnswerTool, Tool, tool
    return (
        CodeAgent,
        DuckDuckGoSearchTool,
        FinalAnswerTool,
        HfApiModel,
        Tool,
        VisitWebpageTool,
        tool,
    )


if __name__ == "__main__":
    app.run()

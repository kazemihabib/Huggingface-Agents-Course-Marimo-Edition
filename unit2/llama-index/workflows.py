# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==3.4.0",
#     "huggingface-hub==0.29.3",
#     "llama-index==0.12.24",
#     "llama-index-callbacks-arize-phoenix==0.4.0",
#     "llama-index-llms-huggingface-api==0.4.1",
#     "llama-index-utils-workflow==0.3.0",
#     "llama-index-vector-stores-chroma==0.4.1",
#     "marimo",
#     "python-dotenv==1.0.1",
#     "pyvis==0.3.2",
# ]
# ///

import marimo

__generated_with = "0.11.20"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Workflows in LlamaIndex


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
    # !pip install llama-index datasets llama-index-callbacks-arize-phoenix llama-index-vector-stores-chroma llama-index-utils-workflow llama-index-llms-huggingface-api pyvis -U -q
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
        ## Basic Workflow Creation

        We can start by creating a simple workflow. We use the `StartEvent` and `StopEvent` classes to define the start and stop of the workflow.
        """
    )
    return


@app.cell
async def _():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


    class _MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            # do something here
            return StopEvent(result="Hello, world!")


    _w = _MyWorkflow(timeout=10, verbose=False)
    _result = await _w.run()
    _result
    return StartEvent, StopEvent, Workflow, step


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Connecting Multiple Steps

        We can also create multi-step workflows. Here we pass the event information between steps. Note that we can use type hinting to specify the event type and the flow of the workflow.
        """
    )
    return


@app.cell
async def _(StartEvent, StopEvent, Workflow, step):
    from llama_index.core.workflow import Event

    class _ProcessingEvent(Event):
        intermediate_result: str

    class _MultiStepWorkflow(Workflow):

        @step
        async def step_one(self, ev: StartEvent) -> _ProcessingEvent:
            return _ProcessingEvent(intermediate_result='Step 1 complete')

        @step
        async def step_two(self, ev: _ProcessingEvent) -> StopEvent:
            final_result = f'Finished processing: {ev.intermediate_result}'
            return StopEvent(result=final_result)
    _w = _MultiStepWorkflow(timeout=10, verbose=False)
    _result = await _w.run()
    _result
    return (Event,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Loops and Branches

        We can also use type hinting to create branches and loops. Note that we can use the `|` operator to specify that the step can return multiple types.
        """
    )
    return


@app.cell
async def _(Event, StartEvent, StopEvent, Workflow, step):
    import random


    class _ProcessingEvent(Event):
        intermediate_result: str


    class _LoopEvent(Event):
        loop_output: str


    class _MultiStepWorkflow(Workflow):
        @step
        async def step_one( self, ev: StartEvent | _LoopEvent) -> _ProcessingEvent | _LoopEvent:
            if random.randint(0, 1) == 0:
                print("Bad thing happened")
                return _LoopEvent(loop_output="Back to step one.")
            else:
                print("Good thing happened")
                return _ProcessingEvent(intermediate_result="First step complete.")

        @step
        async def step_two(self, ev: _ProcessingEvent) -> StopEvent:
            final_result = f"Finished processing: {ev.intermediate_result}"
            return StopEvent(result=final_result)


    w = _MultiStepWorkflow(verbose=False)
    _result = await w.run()
    _result
    return random, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Drawing Workflows

        We can also draw workflows using the `draw_all_possible_flows` function.
        """
    )
    return


@app.cell
def _(__file__, mo, w):
    from llama_index.utils.workflow import draw_all_possible_flows
    from pathlib import Path
    file_path = Path(__file__).parent / "workflow.html"
    draw_all_possible_flows(w, filename=str(file_path.absolute()))
    mo.iframe(file_path.read_text())
    return Path, draw_all_possible_flows, file_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![drawing](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/workflow-draw.png)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### State Management

        Instead of passing the event information between steps, we can use the `Context` type hint to pass information between steps. 
        This might be useful for long running workflows, where you want to store information between steps.
        """
    )
    return


@app.cell
async def _(Event, StartEvent, StopEvent, Workflow, step):
    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import ReActAgent

    class _ProcessingEvent(Event):
        intermediate_result: str

    class _MultiStepWorkflow(Workflow):

        @step
        async def step_one(self, ev: StartEvent, ctx: Context) -> _ProcessingEvent:
            await ctx.set('query', 'What is the capital of France?')
            return _ProcessingEvent(intermediate_result='Step 1 complete')

        @step
        async def step_two(self, ev: _ProcessingEvent, ctx: Context) -> StopEvent:
            query = await ctx.get('query')
            print(f'Query: {query}')
            final_result = f'Finished processing: {ev.intermediate_result}'
            return StopEvent(result=final_result)
    _w = _MultiStepWorkflow(timeout=10, verbose=False)
    _result = await _w.run()
    _result
    return Context, ReActAgent


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multi-Agent Workflows

        We can also create multi-agent workflows. Here we define two agents, one that multiplies two integers and one that adds two integers.
        """
    )
    return


@app.cell
async def _(ReActAgent):
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.core.agent.workflow import AgentWorkflow


    # Define some tools
    def _add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def _multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    _llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    # we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
    _multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="A helpful assistant that can use a tool to multiply numbers.",
        tools=[_multiply], 
        llm=_llm,
    )

    _addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[_add], 
        llm=_llm,
    )

    # Create the workflow
    _workflow = AgentWorkflow(
        agents=[_multiply_agent, _addition_agent],
        root_agent="multiply_agent"
    )

    # Run the system
    _response = await _workflow.run(user_msg="Can you add 5 and 3?")
    _response
    return AgentWorkflow, HuggingFaceInferenceAPI


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

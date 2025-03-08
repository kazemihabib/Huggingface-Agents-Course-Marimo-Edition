# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==0.29.2",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(
    width="full",
    app_title="Dummy Agent Library",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Dummy Agent Library

        In this simple example, **we're going to code an Agent from scratch**.

        This notebook is part of the <a href="https://www.hf.co/learn/agents-course">Hugging Face Agents Course</a>, a free Course from beginner to expert, where you learn to build Agents.

        <img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/share.png" alt="Agent Course"/>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Serverless API

        In the Hugging Face ecosystem, there is a convenient feature called Serverless API that allows you to easily run inference on many models. There's no installation or deployment required.

        To run this notebook, **you need a Hugging Face token** that you can get from https://hf.co/settings/tokens. If you are running this notebook on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN".

        You also need to request access to [the Meta Llama models](meta-llama/Llama-3.2-3B-Instruct), if you haven't done it before. Approval usually takes up to an hour.
        """
    )
    return


@app.cell
def _():
    import os
    from huggingface_hub import InferenceClient

    # os.environ["HF_TOKEN"]="hf_xxxxxxxxxxx"

    # client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
    # if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.2-3B-Instruct
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")
    return InferenceClient, client, os


@app.cell
def _(client):
    # As seen in the LLM section, if we just do decoding, **the model will only stop when it predicts an EOS token**, 
    # and this does not happen here because this is a conversational (chat) model and we didn't apply the chat template it expects.
    output = client.text_generation(
        "The capital of france is",
        max_new_tokens=100,
    )

    print(output)
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As seen in the LLM section, if we just do decoding, **the model will only stop when it predicts an EOS token**, and this does not happen here because this is a conversational (chat) model and **we didn't apply the chat template it expects**.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we now add the special tokens related to the <a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">Llama-3.2-3B-Instruct model</a> that we're using, the behavior changes and it now produces the expected EOS.""")
    return


@app.cell
def _(client):
    prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe capital of france is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    output_1 = client.text_generation(prompt, max_new_tokens=100)
    print(output_1)
    return output_1, prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Using the "chat" method is a much more convenient and reliable way to apply chat templates:""")
    return


@app.cell
def _(client):
    output_2 = client.chat.completions.create(messages=[{'role': 'user', 'content': 'The capital of france is'}], stream=False, max_tokens=1024)
    print(output_2.choices[0].message.content)
    return (output_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The chat method is the RECOMMENDED method to use in order to ensure a **smooth transition between models but since this notebook is only educational**, we will keep using the "text_generation" method to understand the details.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dummy Agent

        In the previous sections, we saw that the **core of an agent library is to append information in the system prompt**.

        This system prompt is a bit more complex than the one we saw earlier, but it already contains:

        1. **Information about the tools**
        2. **Cycle instructions** (Thought → Action → Observation)
        """
    )
    return


@app.cell
def _():
    # This system prompt is a bit more complex and actually contains the function description already appended.
    # Here we suppose that the textual description of the tools has already been appended
    SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

    get_weather: Get the current weather in a given location

    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

    The only values that should be in the "action" field are:
    get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
    example use :
    ```
    {{
      "action": "get_weather",
      "action_input": {"location": "New York"}
    }}

    ALWAYS use the following format:

    Question: the input question you must answer
    Thought: you should always think about one action to take. Only one action at a time in this format:
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: the result of the action. This Observation is unique, complete, and the source of truth.
    ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

    You must always end your output with the following format:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """
    return (SYSTEM_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since we are running the "text_generation" method, we need to add the right special tokens.""")
    return


@app.cell
def _(SYSTEM_PROMPT):
    prompt_1 = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nWhat's the weather in London ?\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return (prompt_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is equivalent to the following code that happens inside the chat method :
        ```
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What's the weather in London ?"},
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

        tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The prompt is now:""")
    return


@app.cell
def _(prompt_1):
    print(prompt_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let’s decode!""")
    return


@app.cell
def _(client, prompt_1):
    output_3 = client.text_generation(prompt_1, max_new_tokens=200)
    # mo.md(output_3)

    print(output_3)
    return (output_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Do you see the problem? 

        The **answer was hallucinated by the model**. We need to stop to actually execute the function!
        """
    )
    return


@app.cell
def _(client, prompt_1):
    output_4 = client.text_generation(prompt_1, max_new_tokens=200, stop=['Observation:'])
    print(output_4)
    return (output_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Much Better!

        Let's now create a **dummy get weather function**. In real situation you could call an API.
        """
    )
    return


@app.cell
def _():
    # Dummy function
    def get_weather(location):
        return f"the weather in {location} is sunny with low temperatures. \n"

    print(get_weather('London'))
    return (get_weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's concatenate the base prompt, the completion until function execution and the result of the function as an Observation and resume the generation.""")
    return


@app.cell
def _(get_weather, output_4, prompt_1):
    new_prompt = prompt_1 + output_4 + get_weather('London')
    print(new_prompt)
    return (new_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is the new prompt:""")
    return


@app.cell
def _(client, new_prompt):
    final_output = client.text_generation(
        new_prompt,
        max_new_tokens=200,
    )

    print(final_output)
    return (final_output,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

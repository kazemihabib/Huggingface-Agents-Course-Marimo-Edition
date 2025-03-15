# 🤗 HuggingFace Agents Course: Marimo Notebook Edition

This repository transforms the popular [HuggingFace Agents course](https://github.com/huggingface/agents-course) notebooks into [Marimo](https://marimo.io) notebooks. Marimo is a reactive Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.

## 📝 Motivation

While working through the HuggingFace Agents course,
I decided to convert the original Jupyter notebooks into Marimo format.
This process serves as both a learning exercise to master Marimo's capabilities
and creates a resource for others interested in experiencing the course content
through a reactive and maintainable notebook environment.

## 🌟 Why Marimo

Marimo is a next-generation notebook environment that addresses many limitations of traditional Jupyter notebooks:

### Problems with Jupyter that Marimo solves

- **Reproducibility:**
  - In Jupyter, hidden state (variables that remain in memory even when their defining code is deleted) and arbitrary execution order lead to notebooks that don't reliably rerun. Marimo tracks dependencies and automatically reruns affected cells.
  - Marimo makes it possible to create and share standalone notebooks, without shipping `requirements.txt` files alongside them (Powered by [uv](https://docs.astral.sh/uv/)). This provides two key benefits:
    - Notebooks that carry their own dependencies are easy to share — just send the .py file!
    - Isolating a notebook from other installed packages prevents obscure bugs arising from environment pollution.
- **Maintainability:** Marimo notebooks are pure Python files, making version control and collaboration straightforward compared to Jupyter's JSON format.

- **Interactivity:** Marimo provides seamless UI components that automatically sync with your Python code, creating a reactive programming experience.

- **Reusability:** Since they're stored as .py files, Marimo notebooks can be executed directly from the command line or imported into other Python programs.

- **Shareability:** Every Marimo notebook can function as an interactive web app with minimal effort, making sharing results simple.

## 🚀 Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Modern Python package installer)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/kazemihabib/hf_agents-course-marimo.git
   cd agents-course-marimo
   ```

2. Use uv to set up the environment and install dependencies:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Run a marimo notebook with sandbox mode:

   ```bash
   marimo edit --sandbox path/to/notebook.py
   ```

   For example:

   ```bash
   marimo edit --sandbox unit2/smolagents/code_agents.py
   ```

   The sandbox mode creates an isolated environment with exactly the dependencies specified in the notebook, ensuring reproducibility.



## 📋 Conversion Progress

Below is a checklist of notebooks from the original HuggingFace Agents course and their conversion status:

- [x] dummy_agent_library.ipynb --> dummy_agent_library.py

### Unit 2: Building Blocks

- Building with SmoL Agents
  - [x] code_agents.ipynb --> code_agents.py
  - [x] tools.ipynb --> tools.py
  - [x] tool_calling_agents.ipynb --> tool_calling_agents.py
  - [x] retrieval_agents.ipynb --> retrieval_agents.py
  - [x] vision_agents.ipynb --> vision_agents.py
  - [x] multiagent_notebook.ipynb --> multiagent.py
  - [x] vision_web_browser.py --> vision_web_browser_marimo.py
- Building with LlamaIndex
  - [ ] components.ipynb --> components.py
  - [ ] tools.ipynb --> tools.py
  - [ ] agents.ipynb --> agents.py
  - [ ] workflows.ipynb --> workflows.py

### Bonus Units

- [ ] bonus-unit1.ipynb --> fine_tuning_function_calling.py

## ⚠️ Compatibility Notes

While adapting the course material to Marimo format, I encountered a few platform-specific differences that required adjustments:

### 1. Authentication with Hugging Face Hub

The authentication method used in Jupyter notebooks doesn't work in Marimo:

```python
# This won't work in Marimo:
from huggingface_hub import notebook_login
notebook_login()
```

This is because Marimo doesn't support ipywidgets, which `notebook_login()` depends on. Ideally, HuggingFace should migrate to anywidget for better compatibility with modern notebook environments like Marimo.

**Workaround:** I used the token-based login approach instead:

```python
from huggingface_hub import login
import os

# Set your token as an environment variable
login(os.environ["HF_TOKEN"])
```

### 2. Issues with `push_to_hub` Function

The `push_to_hub` function fails in Marimo notebooks because it can't extract the tool source codes.

**Workaround:** I created a monkey patch in [patches/smolagents_patches.py](patches/smolagents_patches.py)
 that fixes source code extraction for Marimo environments, enabling Hub uploads.

**Limitation:**  `push_to_hub` doesn't work with local-to-cell tool functions and tool classes (those prefixed with underscore in Marimo).
 For technical details on why this decision has been made, check the Notes section in the patch file.

**Note:** Notebooks that require `push_to_hub` functionality already have this patch applied.
 To use it in other notebooks, add the following code:

```python
import os
import sys
from smolagents import CodeAgent, HfApiModel, Tool, tool, load_tool

# Monkey patch smolagents library to make push_to_hub work with Marimo
notebook_dir = os.path.dirname(os.path.abspath(__file__))  # Get notebook directory
patches_dir = os.path.abspath(os.path.join(notebook_dir, '../../patches'))
sys.path.append(patches_dir)

from smolagents_patches import monekey_patched_get_source
import smolagents.tools
import smolagents.tool_validation
smolagents.tools.get_source = monekey_patched_get_source
smolagents.tool_validation.get_source = monekey_patched_get_source
```

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements:

## 📚 Related Resources

- [HuggingFace Agents Course](https://github.com/huggingface/agents-course)
- [Marimo Documentation](https://docs.marimo.io/)
- [uv Documentation](https://docs.astral.sh/uv/)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

*This project is not officially affiliated with HuggingFace or Marimo.*
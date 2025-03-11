"""
Patching SmolAgents to support Marimo.

This module provides functionality to extract source code from objects in various
Python environments, particularly addressing the challenges in notebook contexts
where standard inspection methods may fail. Specifically, this module adds support
for Marimo notebooks that is not available in the original SmolAgents library.
"""

import textwrap
import inspect
import ast
from pathlib import Path

def _is_marimo() -> bool:  # pragma: no cover
    """Check if we're running in a marimo notebook."""
    try:
        import marimo

        return marimo.running_in_notebook() is True
    except (ImportError, AttributeError):
        return False

def _is_ipython() -> bool:  # pragma: no cover
    """Check if we're running in an IPython environment."""
    try:
        import IPython
        return IPython.get_ipython() is not None
    except (ImportError, AttributeError):
        return False

def monekey_patched_get_source(obj) -> str:
    """Get the source code of a class or callable object (e.g.: function, method).
    First attempts to get the source code using `inspect.getsource`.
    In a dynamic environment (e.g.: Jupyter, IPython), if this fails,
    falls back to retrieving the source code from the current interactive shell session.

    Args:
        obj: A class or callable object (e.g.: function, method)

    Returns:
        str: The source code of the object, dedented and stripped

    Raises:
        TypeError: If object is not a class or callable
        OSError: If source code cannot be retrieved from any source
        ValueError: If source cannot be found in IPython history

    Note:
        TODO: handle Python standard REPL
    """


    #  Notes:
    #  In Mrimo environment:
    #  1) If obj is a Class, `inspect.getsource(obj)` will raise an error
    #  2) If obj is a function, `inspect.getsource(obj)` will return the source code
    #  3) If obj is a Class, in marimo `inspect.getfile(obj)` will return the path of the file
    #  4) If obj is a function, in marimo `inspect.getfile(obj)` will return a non-useful path
    #  5) If obj is local to cell (_ prefixed) `obj.__name__` will return `_cell_{cell_id}{obj orig name}`
    #  so if obj is a function, the smolagents `to_dict` function
    #  (https://github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py#L243)
    #  attempts to replace the function name with `forward` in the source code, but fails because it can't 
    #  find the prefixed name pattern. As a result, the uploaded tool lacks a `forward` function
    #  and will fail when executed on the server.
    #  6) Due to notes 4 and 5, local to cell functions cannot use this feature. Even though this could
    #  be fixed by modifying the smolagents tools file, note 1 requires us to read the entire
    #  source file to extract class definitions. This creates ambiguity when multiple local to cell
    #  classes have the same name. For consistency, neither local functions nor local classes
    #  are allowed to use this feature.
    
    


    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")
    
    if _is_marimo():
        # TODO("Probably not a good place to check this, but I don't want to monkey patch more modules,
        # check this in a more proper place in the pull-request to smolagents.")
        if obj.__name__.startswith("_"):
            # Check Note 6
            raise ValueError(
    "Cannot extract source code from local to cell objects (prefixed with underscore) in Marimo environment. "
    "Please define your tool function or tool class without underscore prefix."
)

    inspect_error = None
    try:
        return textwrap.dedent(inspect.getsource(obj)).strip()
    except OSError as e:
        # let's keep track of the exception to raise it if all further methods fail
        inspect_error = e
        
    if _is_marimo():
        try:
            file = inspect.getfile(obj)
            all_cells = Path(file).read_text()
            
            tree = ast.parse(all_cells)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                    return textwrap.dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        except (OSError, TypeError) as e:
            # Marimo is available but we couldn't find the source code, let's raise the error
            raise e from inspect_error
            
    if _is_ipython():
        import IPython

        shell = IPython.get_ipython()
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return textwrap.dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        # If IPython approach fails, pass the error to be raised with the original inspect error
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history") from inspect_error
    
    # If we reach here, all methods failed
    raise inspect_error 
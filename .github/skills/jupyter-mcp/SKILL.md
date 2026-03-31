---
name: jupyter-mcp
description: >
  Guide for operating Jupyter notebooks via the JupyterMCP MCP server. Use this skill
  whenever working with Jupyter notebooks, running code in cells, reading or modifying
  notebook content, managing kernels, or executing any analysis in Jupyter — even if the
  user just says "open a notebook", "run this code", "check the output", or "add a cell".
  This skill is the foundation for all notebook-based work; load it alongside
  notebook-data-exploration or notebook-ml when doing data science tasks.
---

# JupyterMCP — Operating Jupyter Notebooks

This skill teaches you how to use the JupyterMCP tools to work with live Jupyter notebooks.
The JupyterMCP server connects to a running Jupyter Lab instance and exposes tools for
reading, editing, and executing notebook cells.

> **Before anything else:** Jupyter Lab must already be running. In this project, launch it
> with `pixi run open-jupyter-lab` from a pixi shell. The MCP server connects automatically.

---

## Tool Reference

### Connection & Session

| Tool | When to use |
|------|-------------|
| `JupyterMCP-connect_to_jupyter` | Connect to a non-default Jupyter server (different port/token). Usually not needed — the default connection works. |
| `JupyterMCP-use_notebook` | **Always call first.** Activates a notebook for all subsequent cell operations. Use `mode: "connect"` for existing notebooks, `mode: "create"` for new ones. |
| `JupyterMCP-list_notebooks` | See which notebooks you've already activated in this session. |
| `JupyterMCP-unuse_notebook` | Release a notebook when done to free resources. |
| `JupyterMCP-restart_notebook` | Restart the kernel (clears all variables). Use when the kernel is stuck or you need a clean slate. |

### Navigation & Reading

| Tool | When to use |
|------|-------------|
| `JupyterMCP-list_files` | Browse the Jupyter server's file system to find notebooks. |
| `JupyterMCP-list_kernels` | Check available kernels and their status. |
| `JupyterMCP-read_notebook` | Read cell contents. Use `response_format: "brief"` first to get the overview (cell index, type, first line); then `"detailed"` with `start_index` + `limit` to read specific cells. |
| `JupyterMCP-read_cell` | Read a single cell including its outputs. Good for inspecting a specific cell's result without loading the whole notebook. |

### Editing

| Tool | When to use |
|------|-------------|
| `JupyterMCP-insert_cell` | Add a new cell at a specific index. Use `cell_index: -1` to append at the end. |
| `JupyterMCP-overwrite_cell_source` | Replace the code/markdown in an existing cell. Returns a diff so you can verify the change. |
| `JupyterMCP-delete_cell` | Remove one or more cells by index. |

### Execution

| Tool | When to use |
|------|-------------|
| `JupyterMCP-execute_cell` | Execute a specific cell by index. Use for cells that are already in the notebook. |
| `JupyterMCP-insert_execute_code_cell` | **The main workhorse.** Insert a new code cell AND execute it in one step. Saves the cell permanently in the notebook. Use for analysis, exploration, and any work you want to keep. |
| `JupyterMCP-execute_code` | Run ephemeral code that is **NOT saved** to the notebook. Ideal for magic commands (`%pip install`, `%timeit`), quick variable inspection (`df.head()`), or temporary debugging. |

---

## Core Workflows

### 1. Starting work on an existing notebook

```
1. JupyterMCP-list_files        → find the notebook path
2. JupyterMCP-use_notebook      → activate it (mode: "connect")
3. JupyterMCP-read_notebook     → brief format, limit 30 — get the lay of the land
4. JupyterMCP-read_cell         → inspect specific cells with outputs if needed
5. Begin editing or executing
```

### 2. Creating a new notebook

```
1. JupyterMCP-use_notebook      → mode: "create", choose path under notebooks/
2. JupyterMCP-insert_execute_code_cell → add imports and setup as first cell
3. Continue building cells
```

### 3. Executing analysis

Prefer `insert_execute_code_cell` over separate insert + execute calls — it's one round-trip
and automatically saves the code. Use `execute_code` only for throwaway snippets.

```python
# Example: insert_execute_code_cell to add a persistent analysis cell
cell_source = """
import polars as pl
df = pl.read_csv('../data/landing/mens/MNCAATourneyCompactResults.csv')
print(df.shape)
df.head()
"""
```

### 4. Editing existing cells

When you need to fix or improve a cell:
1. `read_notebook` (brief) to locate the cell index
2. `read_cell` to see current source and outputs
3. `overwrite_cell_source` with the corrected code
4. `execute_cell` to re-run it

---

## Key Distinctions

**`execute_code` vs `insert_execute_code_cell`**

Think of `execute_code` as your scratchpad — output appears but nothing is saved to the
notebook. Use it to inspect variables, check shapes, run pip installs, or profile code.

`insert_execute_code_cell` is for work you want to keep. It becomes part of the notebook's
permanent record and can be re-run later.

**`read_notebook` brief vs detailed**

Always start with `brief` to get the structure. Only switch to `detailed` for specific cells
you need to read fully — loading the entire notebook in detailed mode on a large notebook is
slow and context-heavy.

---

## Error Handling

**Kernel died / unresponsive:** Call `JupyterMCP-restart_notebook`. Note that this clears all
in-memory variables — you'll need to re-run earlier cells to restore state.

**Cell execution timeout:** `execute_cell` and `insert_execute_code_cell` have a configurable
`timeout` parameter (default 90s). For long-running training jobs, increase this to 300-600s
or use `stream: true` with a `progress_interval` to get live updates.

**"Notebook not activated" errors:** You must call `use_notebook` before any cell operation.
If you switched notebooks, call `use_notebook` again to reactivate.

**Wrong cell index after inserts/deletes:** Cell indices shift when you insert or delete.
Always `read_notebook` (brief) to refresh indices before operating on specific cells.

---

## Project Conventions (mm2026)

- **Notebooks live in** `notebooks/` — use `notebooks/crappy/` for throwaway experiments
- **Naming convention:** prefix with initials, e.g. `jl.feature_engineering.ipynb`
- **Kernel:** always use the default `Python 3` kernel (no special registration needed)
- **Data paths:** always relative — `../data/landing/`, `../data/landing/mens/`, `../data/landing/womens/`
- **Environment:** Jupyter runs inside the Pixi environment; all project packages are available

---

## Related Skills

- **notebook-data-exploration** — EDA workflows, data profiling, visualization patterns
- **notebook-ml** — Feature engineering, model building, competition submission

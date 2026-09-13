# Agentic Totchef
![logo](static/logo.svg)

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/org-carmelolg-private-labs/agentic-totchef)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Coverage](https://raw.githubusercontent.com/org-carmelolg-private-labs/agentic-totchef/master/static/coverage.svg)](https://github.com/org-carmelolg-private-labs/agentic-totchef/actions)

Agentic Totchef is a small, local-first example that demonstrates how to build generative AI agents that manage recipes and kindergarten menus using Ollama for reasoning and tool calls. The project is intentionally lightweight and modular so it can be used as a learning reference or a base for experimentation.

## Key features

- LangGraph-based batch workflow (`StateGraph`) orchestrating menu generation, merging and shopping-list steps, with kindergarten/home menus generated in parallel.
- A LangGraph ReAct agent (`langchain.agents.create_agent`) for the chatbot, with proper multi-round tool-calling and cross-turn conversation memory (`InMemorySaver`, scoped per session/browser tab).
- Domain tools for Home Kitchen and Kindergarten workflows (HTTP-based tools), exposed as LangChain `@tool`-decorated functions.
- Chroma-backed (in-memory, no server) semantic matching for fuzzy category lookups, embedding with a local Ollama model.
- 100% local/free: Ollama for both chat and embedding models, Chroma runs embedded — no cloud APIs or paid services required.
- Verbose tracing and error handling (incl. structured logging for HTTP fallbacks) to aid development and debugging.

## Prerequisites

- Python 3.10+
- Ollama installed and running (e.g. run `ollama serve`)
- Ollama models for reasoning (e.g. `qwen3:latest`) and embeddings (e.g. `nomic-embed-text:latest`)
- A `.env` file based on `.env.example` with required configuration values

## Installation

```bash
git clone https://github.com/org-carmelolg-private-labs/agentic-totchef.git
cd agentic-totchef
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to set your models and hosts
```

## Configuration

Set the important variables in `.env` (examples):

```
EMBEDDING_MODEL=nomic-embed-text:latest
LANGUAGE_MODEL=qwen3:latest
THINKING_MODE=True
LLM_PROVIDER=ollama
# Optional: Ollama server base URL. If not set, defaults to http://localhost:11434
OLLAMA_HOST=http://localhost:11434

# Optional HTTP service hosts used by the domain tools
# If not set, defaults are under static folder under .json files
HOME_KITCHEN_API_HOST=http://localhost:8000/kitchen
KINDERGARTEN_API_HOST=http://localhost:8000/kindergarten
```

## Getting Started

- Ensure Ollama is running and the models referenced in `.env` are available locally.
- Activate the virtual environment and install dependencies as shown above.
- Start a chat session or run the batch runner to exercise the agent behaviour.

### Usage

1. Quick interactive (CLI):
```terminal
python agentic-totchef-chat.py
``` 
starts a REPL-like chat interface where the agent reasons and can call tools.

2. Quick interactive (GUI):
```terminal
python agentic-totchef-gui-chat.py
``` 
launches the GUI chat (usually on http://localhost:8080).

3. Batch generation:
```terminal
python agentic-totchef.py
``` 
runs the automatic menu-generation flow (useful for scheduled or CI runs).

### Examples

- Generate a full-week menu (batch):

```bash
python agentic-totchef.py
```

- Ask the chat agent for domain information (CLI or GUI):

```
> Give me the home kitchen vegetables available
> What is Tuesday's menu for week 2 from the kindergarten data?
> Create a healthy weekly menu plan based on the kindergarten week 1 menu
```

## Running with Docker

Build and run the container:

```bash
docker build -t agentic-totchef .
docker run -d -p 8080:8080 --name agentic-totchef agentic-totchef
```

## License

This project is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0). See `LICENSE.md` for details.

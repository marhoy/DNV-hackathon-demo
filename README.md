# Overview

This repository contains code for demonstrating LLM applications using LangChain and the
LangServe app with templates. It was written for the 2024 digital trainee hackathon at
DNV.

This repository already contains a few
[templates](https://python.langchain.com/docs/templates/) from the [LangChain
repository](https://github.com/langchain-ai/langchain/blob/master/templates/README.md).
You can add more templates by following the instructions below.

# How to set up a development environment

## Standard Python prerequisites

- Install Python >= 3.11, either from the [official website](https://www.python.org/downloads/) or by using `pyenv` (recommended).
- Install poetry by following the instructions on the [official website](https://python-poetry.org/docs/). Installation with `pipx` is recommended.
- Create a new virtual environment for this project in your favourite way. E.g. with `pyenv`, `poetry` or `venv`.

## Providing secrets

Create a file named e.g. `secrets.env` in the root of the project and add the following content:

```shell
# Required: OpenAI API key
OPENAI_API_KEY=<your-openai-api-key>

# Optional: If you want to use LangSmith for tracing / debugging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-api-key>
LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you want to make the environment variables available to a specific command, you can
use `dotenv` like this:

```bash
dotenv -f secrets.env run <your-command>
```

Or if you want to export all the environment variables in the file to your shell, you can
do this:

```bash
export $(cat secrets.env | xargs)
```

## Installing all dependencies

Run the following command to install the project dependencies:

```bash
poetry install
```

# Notebooks with example code

I the `notebooks` directory, you will find a few Jupyter notebooks with example code for
understanding the use of Embeddings and LLMs in general.

# Launching the LangServe app

Load secrets from the .env-file and start the LangServe app. Optionally change the port
number to something else.

```bash
dotenv -f secrets.env run uvicorn app.server:app --host 0.0.0.0 --port 8765 --reload
```

# Adding more langchain template packages

## Adding a package

```bash
# Add packages from
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add --no-pip <package-name>

# Install the package and its dependencies in the virtual environment
poetry update
```

> [!NOTE]  
>  After adding a package, you need to update the file `app/server.py` as instructed in
> the output of the command.

## Removing a package

```bash
# Remove the package and its dependencies from the environment
poetry remove <package-name>

# Remove the files from the `packages` directory
langchain app remove <package-name>
```

# Running the app in a Docker container

Build the Docker image:

```bash
docker build . -t my-langserve-app
```

Run the Docker container. Inside the container, uvicorn will listen to port 80. We map
that to whatever port we want to use (the example below uses port 8765).

```bash
docker run --rm -it -p 8765:80 --env-file secrets.env my-langserve-app
```

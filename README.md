# Setting up a development environment

## Standard Python prerequisites

- Install Python >= 3.11, either from the [official website](https://www.python.org/downloads/) or by using `pyenv` (recommended).
- Install poetry by following the instructions on the [official website](https://python-poetry.org/docs/). Installation with `pipx` is recommended.
- Create a new virtual environment for this project in your favourite way. E.g. with `pyenv`, `poetry` or `venv`.

## Providing secrets

Create a file named e.g. `secrets.env` in the root of the project and add the following content:

```shell
# Required: OpenAI API key
export OPENAI_API_KEY=<your-openai-api-key>

# Optional: If you want to use LangSmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Installing all dependencies

Run the following command to install the project dependencies:

```bash
poetry install
```

# Launching the LangServe app

Load secrets from the .env-file and start the LangServe app:

```bash
dotenv -f secrets.env run uvicorn app.server:app --host 0.0.0.0 --port 8765 --reload
```

# Adding more langchain template packages

## Adding a package

```bash
# adding packages from
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add --no-pip <package-name>

# Install the package in the environment
poetry update
```

> [!NOTE]  
>  After adding a package, you need to update the file `app/server.py` as instructed in
> the output of the command.

## Removing a package

```bash
# Remove the package from the environment
poetry remove <package-name>

# Remove the files from the `packages` directory
langchain app remove <package-name>
```

# Running the app in a Docker container

Build the Docker image:

```bash
docker build . -t my-langserve-app
```

Running the Docker container. Inside the container, uvicorn will listen to port 80.
Map that to whatever port you want to use (the example below uses port 8765).

```bash
docker run --rm -it -p 8765:80 --env-file secrets.env my-langserve-app
```

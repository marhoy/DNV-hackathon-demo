FROM python:3.12-slim-bookworm

# Define some environment variables
ENV PIP_NO_CACHE_DIR=true \
    DEBIAN_FRONTEND=noninteractive

# Install system deb-packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    pipx \
    authbind

# We want to run things as a non-privileged user
ENV USERNAME=user
ENV PATH="$PATH:/home/$USERNAME/.local/bin:/home/$USERNAME/app/.venv/bin"

# Add user and set up a workdir
RUN useradd -m $USERNAME -u 12345
WORKDIR /home/$USERNAME/app
RUN chown $USERNAME.$USERNAME .

# Allow unprivileged user to run listen on port 80
RUN touch /etc/authbind/byport/80
RUN chmod 500 /etc/authbind/byport/80
RUN chown $USERNAME /etc/authbind/byport/80

# Everything below here runs as a non-privileged user
USER $USERNAME

# Install poetry
RUN pipx install poetry==1.8.2
RUN poetry config virtualenvs.in-project true

# Install runtime dependencies (will be cached)
COPY pyproject.toml poetry.lock ./
COPY packages ./packages
RUN poetry install --without dev --no-root

# Copy the rest of the project files to container
COPY . .
RUN poetry install --without dev

# Start the server
EXPOSE 80
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "80"]

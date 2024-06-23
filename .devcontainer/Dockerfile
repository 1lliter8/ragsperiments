# See here for image contents: https://github.com/devcontainers/images/blob/c0887a58a0d7dca45f86ba8da94cafc222b52b58/src/python/.devcontainer/Dockerfile

ARG VARIANT="1-3.12"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

USER vscode

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Update Debian
RUN \
  sudo apt-get -y update \
  && sudo apt-get install -y tzdata

# Install Python and dependecies
RUN \
  pipx install poetry pre-commit poethepoet \
  && poetry completions bash >> ~/.bash_completion

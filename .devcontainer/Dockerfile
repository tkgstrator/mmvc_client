FROM mcr.microsoft.com/devcontainers/python:dev-3.12-bookworm

RUN \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y espeak gosu libsndfile1-dev

WORKDIR /home/vscode/
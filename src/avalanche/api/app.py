"""The FastAPI application skeleton."""

from fastapi import FastAPI

from avalanche.config import ResolvedConfig

app = FastAPI(title="avalanche")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config-options")
def config_options() -> dict[str, object]:
    return {"schema": ResolvedConfig.model_json_schema()}

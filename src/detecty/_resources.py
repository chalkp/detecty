"""Resolve packaged default config files (importlib.resources)."""
from importlib.resources import files


def data_path(name: str) -> str:
    """Absolute path to a file shipped in detecty/data/ (e.g. 'config.yaml')."""
    return str(files("detecty.data").joinpath(name))


def default_config() -> str:
    return data_path("config.yaml")


def default_ensemble() -> str:
    return data_path("ensemble.yaml")


def default_visual_prompts() -> str:
    return data_path("visual_prompts.yaml")

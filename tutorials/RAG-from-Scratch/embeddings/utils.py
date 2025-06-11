# embeddings/load_all.py
import importlib
import os
import pkgutil


def load_embeddings():
    package_name = "embeddings"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package_name}.{name}")

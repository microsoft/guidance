import os
import json

from pathlib import Path

from ._model import Model
from .llama_cpp._llama_cpp import LlamaCppEngine

base = Path(os.getenv("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
blobs = base / "blobs"
library = base / "manifests" / "registry.ollama.ai" / "library"


class Ollama(Model):
    def __init__(
        self,
        model: str,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        **llama_cpp_kwargs,
    ):
        """Wrapper for models pulled using Ollama.

        Gets the local model path using the provided model name, and
        then instantiates the `LlamaCppEngine` with it and other args.
        """

        manifest = library / Path(model.replace(":", "/") if ":" in model else model + "/latest")

        if not manifest.exists():
            raise ValueError(f"Model '{model}' not found in library.")

        with open(manifest, "r") as f:
            for layer in json.load(f)["layers"]:
                if layer["mediaType"] == "application/vnd.ollama.image.model":
                    digest: str = layer["digest"]
                    break
            else:
                raise ValueError("Model layer not found in manifest.")

        engine = LlamaCppEngine(
            model=(blobs / digest.replace(":", "-")),
            compute_log_probs=compute_log_probs,
            chat_template=chat_template,
            **llama_cpp_kwargs,
        )

        super().__init__(engine, echo=echo)

"""Modelos de lenguaje: carga, interpretación y descripción fiel."""

from scene.llm.describer import DescriptionResult, FidelityReport, SceneDescriber
from scene.llm.interpreter import InterpretationResult, Interpreter
from scene.llm.model import TransformersLLM, build_llm

__all__ = [
    "DescriptionResult",
    "FidelityReport",
    "InterpretationResult",
    "Interpreter",
    "SceneDescriber",
    "TransformersLLM",
    "build_llm",
]

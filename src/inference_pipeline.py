"""Inference utilities for deployed ODS model."""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np

# Imports required for unpickling pipeline steps.
from src.config import (
    BEST_METADATA_PATH,
    BEST_MODEL_PATH,
    DEPLOY_METADATA_PATH,
    DEPLOY_MODEL_PATH,
)
from src.text_features import (  # noqa: F401
    HFEmbeddingTransformer,
    MeanGensimEmbeddingTransformer,
    TextPreprocessor,
)


def _candidate_artifacts(prefer_deploy: bool) -> list[tuple[Any, Any]]:
    if prefer_deploy:
        return [
            (DEPLOY_MODEL_PATH, DEPLOY_METADATA_PATH),
            (BEST_MODEL_PATH, BEST_METADATA_PATH),
        ]
    return [
        (BEST_MODEL_PATH, BEST_METADATA_PATH),
        (DEPLOY_MODEL_PATH, DEPLOY_METADATA_PATH),
    ]


def load_artifacts(prefer_deploy: bool = True) -> tuple[Any, dict[str, Any]]:
    for model_path, metadata_path in _candidate_artifacts(prefer_deploy):
        if model_path.exists() and metadata_path.exists():
            model = joblib.load(model_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["artifact_model_path"] = str(model_path)
            metadata["artifact_metadata_path"] = str(metadata_path)
            return model, metadata

    raise FileNotFoundError(
        "No se encontraron artifacts validos. "
        f"Revise: {DEPLOY_MODEL_PATH}, {DEPLOY_METADATA_PATH}, {BEST_MODEL_PATH}, {BEST_METADATA_PATH}"
    )


def predict_text(text: str, top_k: int = 3, prefer_deploy: bool = True) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("El texto de entrada está vacío.")

    model, metadata = load_artifacts(prefer_deploy=prefer_deploy)

    pred = model.predict([text])[0]
    result: dict[str, Any] = {
        "predicted_ods": int(pred) if str(pred).isdigit() else pred,
        "model_name": metadata.get("winner_experiment", metadata.get("model_name", "unknown")),
        "confidence": None,
        "top_k": [],
        "artifact_model_path": metadata.get("artifact_model_path"),
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
            result["confidence"] = float(np.max(proba))
            classes = getattr(model, "classes_", None)
            if classes is not None:
                pairs = list(zip(classes, proba))
                pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[: max(1, top_k)]
                result["top_k"] = [{"ods": int(c), "prob": float(p)} for c, p in pairs]
        except Exception:
            result["confidence"] = None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia de ODS para texto libre.")
    parser.add_argument("--text", required=True, help="Texto para clasificar.")
    parser.add_argument("--top-k", type=int, default=3, help="Numero de clases top a retornar.")
    parser.add_argument(
        "--prefer-notebook-model",
        action="store_true",
        help="Usar primero best_model.joblib en lugar de deploy_model.joblib.",
    )
    args = parser.parse_args()

    output = predict_text(
        args.text,
        top_k=args.top_k,
        prefer_deploy=not args.prefer_notebook_model,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

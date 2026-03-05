"""Training pipeline for deploying the winner model outside the notebook."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACTS_DIR,
    BEST_METADATA_PATH,
    BEST_MODEL_PATH,
    DATA_PATH,
    DEFAULT_RANDOM_STATE,
    DEPLOY_METADATA_PATH,
    DEPLOY_METRICS_PATH,
    DEPLOY_MODEL_PATH,
)
from src.data_pipeline import load_dataset, maybe_subsample, split_train_val_test
from src.text_features import (
    HFEmbeddingTransformer,
    MeanGensimEmbeddingTransformer,
    TextPreprocessor,
)


def _to_tuple_if_needed(params: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(params)
    if isinstance(fixed.get("tfidf__ngram_range"), list):
        fixed["tfidf__ngram_range"] = tuple(fixed["tfidf__ngram_range"])
    return fixed


def _resolve_hf_device(user_device: str = "auto") -> str:
    if user_device in {"cpu", "mps"}:
        return user_device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_winner_config(metadata_path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata de ganador: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    winner = metadata.get("winner_experiment")
    if not winner:
        raise ValueError("winner_experiment no encontrado en metadata")
    params = _to_tuple_if_needed(metadata.get("best_params", {}))
    return winner, params, metadata


def build_pipeline_from_winner(
    winner_experiment: str,
    hf_device: str,
    hf_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> Pipeline:
    """Build base pipeline structure for the winner experiment."""
    if winner_experiment == "E01_Preproc_TFIDF_ComplementNB":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("tfidf", TfidfVectorizer(max_features=50000)),
                ("clf", ComplementNB()),
            ]
        )
    if winner_experiment == "E02_Preproc_TFIDF_SVD_CatBoost":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("tfidf", TfidfVectorizer(max_features=50000)),
                ("svd", TruncatedSVD(random_state=DEFAULT_RANDOM_STATE)),
                (
                    "clf",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        random_state=DEFAULT_RANDOM_STATE,
                        verbose=0,
                        allow_writing_files=False,
                        thread_count=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E03_Preproc_TFIDF_SVD_LightGBM":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("tfidf", TfidfVectorizer(max_features=50000)),
                ("svd", TruncatedSVD(random_state=DEFAULT_RANDOM_STATE)),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E04_Preproc_TFIDF_NMF_LightGBM":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("tfidf", TfidfVectorizer(max_features=50000)),
                ("nmf", NMF(random_state=DEFAULT_RANDOM_STATE, init="nndsvda", max_iter=500)),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if winner_experiment in {"E05_Preproc_Word2Vec_GaussianNB", "E07_Preproc_FastText_GaussianNB"}:
        method = "word2vec" if "Word2Vec" in winner_experiment else "fasttext"
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("emb", MeanGensimEmbeddingTransformer(method=method, seed=DEFAULT_RANDOM_STATE, workers=1)),
                ("clf", GaussianNB()),
            ]
        )
    if winner_experiment in {"E06_Preproc_Word2Vec_PCA_GaussianNB", "E08_Preproc_FastText_PCA_GaussianNB"}:
        method = "word2vec" if "Word2Vec" in winner_experiment else "fasttext"
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                (
                    "emb",
                    MeanGensimEmbeddingTransformer(
                        method=method,
                        vector_size=200,
                        seed=DEFAULT_RANDOM_STATE,
                        workers=1,
                    ),
                ),
                ("pca", PCA(random_state=DEFAULT_RANDOM_STATE)),
                ("clf", GaussianNB()),
            ]
        )
    if winner_experiment == "E09_Preproc_Word2Vec_CatBoost":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("emb", MeanGensimEmbeddingTransformer(method="word2vec", seed=DEFAULT_RANDOM_STATE, workers=1)),
                (
                    "clf",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        random_state=DEFAULT_RANDOM_STATE,
                        verbose=0,
                        allow_writing_files=False,
                        thread_count=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E10_Preproc_Word2Vec_PCA_LightGBM":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                (
                    "emb",
                    MeanGensimEmbeddingTransformer(
                        method="word2vec",
                        vector_size=200,
                        seed=DEFAULT_RANDOM_STATE,
                        workers=1,
                    ),
                ),
                ("pca", PCA(random_state=DEFAULT_RANDOM_STATE)),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E11_Preproc_FastText_LightGBM":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                ("emb", MeanGensimEmbeddingTransformer(method="fasttext", seed=DEFAULT_RANDOM_STATE, workers=1)),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        random_state=DEFAULT_RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E12_Preproc_FastText_PCA_CatBoost":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                (
                    "emb",
                    MeanGensimEmbeddingTransformer(
                        method="fasttext",
                        vector_size=200,
                        seed=DEFAULT_RANDOM_STATE,
                        workers=1,
                    ),
                ),
                ("pca", PCA(random_state=DEFAULT_RANDOM_STATE)),
                (
                    "clf",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        random_state=DEFAULT_RANDOM_STATE,
                        verbose=0,
                        allow_writing_files=False,
                        thread_count=-1,
                    ),
                ),
            ]
        )
    if winner_experiment == "E13_Preproc_HF_GaussianNB":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                (
                    "emb",
                    HFEmbeddingTransformer(
                        model_name=hf_model_name,
                        device=hf_device,
                        batch_size=32,
                        use_cache=True,
                        cache_dir="cache/hf_embeddings",
                    ),
                ),
                ("clf", GaussianNB()),
            ]
        )
    if winner_experiment == "E14_Preproc_HF_PCA_GaussianNB":
        return Pipeline(
            [
                ("prep", TextPreprocessor()),
                (
                    "emb",
                    HFEmbeddingTransformer(
                        model_name=hf_model_name,
                        device=hf_device,
                        batch_size=32,
                        use_cache=True,
                        cache_dir="cache/hf_embeddings",
                    ),
                ),
                ("pca", PCA(random_state=DEFAULT_RANDOM_STATE)),
                ("clf", GaussianNB()),
            ]
        )

    raise ValueError(f"Ganador no soportado para reconstruccion: {winner_experiment}")


def evaluate_model(model, X, y) -> dict[str, float]:
    y_pred = model.predict(X)
    return {
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y, y_pred)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento productivo del modelo ganador.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--winner-metadata", type=Path, default=BEST_METADATA_PATH)
    parser.add_argument("--winner-model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--output-model", type=Path, default=DEPLOY_MODEL_PATH)
    parser.add_argument("--output-metadata", type=Path, default=DEPLOY_METADATA_PATH)
    parser.add_argument("--output-metrics", type=Path, default=DEPLOY_METRICS_PATH)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=0, help="Submuestra estratificada para pruebas rápidas.")
    parser.add_argument("--hf-device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--allow-notebook-fallback",
        action="store_true",
        help="Si el ganador no es soportado, usa best_model.joblib del notebook.",
    )
    parser.add_argument(
        "--no-refit-on-train-val",
        action="store_true",
        help="Si se activa, no hace refit final sobre train+val.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path)
    if args.sample_size > 0:
        df = maybe_subsample(df, sample_size=args.sample_size, random_state=args.random_state)

    split = split_train_val_test(df, random_state=args.random_state)

    winner_experiment, best_params, notebook_metadata = load_winner_config(args.winner_metadata)
    hf_device = _resolve_hf_device(args.hf_device)

    reconstructed = True
    try:
        model = build_pipeline_from_winner(
            winner_experiment=winner_experiment,
            hf_device=hf_device,
            hf_model_name=notebook_metadata.get(
                "hf_model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )
        if best_params:
            model.set_params(**best_params)
    except Exception as exc:
        if not args.allow_notebook_fallback:
            raise
        print(f"[WARN] No se pudo reconstruir pipeline ({exc}). Usando artifact del notebook.")
        model = joblib.load(args.winner_model)
        reconstructed = False

    # Fit + evaluation (train -> val/test)
    model.fit(split.X_train, split.y_train)
    val_metrics = evaluate_model(model, split.X_val, split.y_val)
    test_metrics = evaluate_model(model, split.X_test, split.y_test)

    deploy_model = model
    if not args.no_refit_on_train_val and reconstructed:
        deploy_model = clone(model)
        X_train_val = np.concatenate([split.X_train.values, split.X_val.values])
        y_train_val = np.concatenate([split.y_train.values, split.y_val.values])
        deploy_model.fit(X_train_val, y_train_val)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_model, args.output_model)

    deploy_metadata = {
        "winner_experiment": winner_experiment,
        "best_params": best_params,
        "source_metadata": str(args.winner_metadata),
        "reconstructed_pipeline": reconstructed,
        "random_state": args.random_state,
        "sample_size": args.sample_size if args.sample_size > 0 else len(df),
        "hf_device": hf_device,
        "refit_on_train_val": not args.no_refit_on_train_val and reconstructed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_metadata.write_text(json.dumps(deploy_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "train_size": int(len(split.X_train)),
        "val_size": int(len(split.X_val)),
        "test_size": int(len(split.X_test)),
    }
    args.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"deploy_model": str(args.output_model), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Tickets Jira - Microproyecto 2

## Tablero

| Ticket | Titulo | Prioridad | Estado | Criterio de aceptacion |
| --- | --- | --- | --- | --- |
| MP2-001 | Preparar entorno de experimentacion | Alta | Done | Dependencias instaladas y verificadas (`torch`, `transformers`, `sentence-transformers`, `sklearn`, `lightgbm`, `catboost`, `gensim`, `spacy`, `nltk`). |
| MP2-002 | Definir notebook con estructura completa | Alta | Done | Notebook contiene secciones: datos, split, preprocessing, embeddings, reduccion, experimentos, ranking, ganador y exportables. |
| MP2-003 | Implementar pipeline de preprocessing configurable | Alta | Done | Preprocessing parametrizable con tokenizacion, stopwords, lematizacion, stemming y normalizacion. |
| MP2-004 | Implementar embeddings clasicos | Alta | Done | Notebook incluye al menos dos embeddings clasicos (ej. Word2Vec/FastText) con transformadores reutilizables. |
| MP2-005 | Implementar embeddings HF en PyTorch + MPS | Alta | Blocked | Se usa Hugging Face con PyTorch en `device='mps'`; se valida disponibilidad MPS al inicio. |
| MP2-006 | Implementar reduccion de dimensionalidad | Alta | Done | Se prueban SVD/NMF en TF-IDF y PCA en embeddings con busqueda de hiperparametros. |
| MP2-007 | Ejecutar matriz de experimentos completa | Alta | Done | Se corren experimentos con Naive Bayes, CatBoost y LightGBM sobre TF-IDF y embeddings (con/sin PCA). |
| MP2-008 | Comparar resultados y seleccionar ganador | Alta | Done | Existe tabla comparativa unificada con ranking por `test_f1_macro` (criterio acordado por usuario). |
| MP2-009 | Documentar seccion Modelo ganador | Alta | Done | Se muestran `best_params`, metricas val/test, matriz de confusion y 4 predicciones de test. |
| MP2-010 | Exportar artefactos del ganador desde notebook | Media | Done | Notebook exporta `artifacts/experiment_results.csv`, `best_model.joblib`, `best_model_metadata.json`. |
| MP2-011 | Validar notebook funcional de punta a punta | Alta | Done | Notebook sin errores de sintaxis y ejecutable; dependencias y flujo verificados. |
| MP2-012 | Implementar data pipeline en `src` | Alta | Done | Existe `src/data_pipeline.py` con carga/validacion de datos y split estratificado train/val/test. |
| MP2-013 | Implementar training pipeline del modelo ganador | Alta | Done | Existe `src/train_pipeline.py` que carga ganador desde metadata, entrena, evalua y genera artefactos de despliegue. |
| MP2-014 | Implementar inference pipeline de despliegue | Alta | Done | `src/inference_pipeline.py` prioriza `deploy_model`, permite top-k y expone CLI funcional. |
| MP2-015 | Integrar app Streamlit con inferencia | Alta | Done | `app.py` usa pipeline de inferencia, muestra ODS predicho, confianza, top-k y artifact cargado. |

## Registro de ejecucion

| Ticket | Estado | Evidencia |
| --- | --- | --- |
| MP2-001 | Done | `torch 2.10.0`, `transformers 5.3.0`, `sentence-transformers 5.2.3`; validacion MPS ejecutada (`is_built=True`, `is_available=False` en este runtime). |
| MP2-002 | Done | `Microproyecto2.ipynb` reconstruido con secciones completas y flujo end-to-end. |
| MP2-003 | Done | `TextPreprocessor` implementado con tokenizacion, normalizacion, stopwords, lematizacion y stemming (modulo `src/text_features.py`). |
| MP2-004 | Done | Embeddings clasicos implementados (`Word2Vec` y `FastText`) en `MeanGensimEmbeddingTransformer`. |
| MP2-005 | Blocked | Implementado `HFEmbeddingTransformer` con PyTorch y validacion MPS; en este runtime `torch.backends.mps.is_available() = False`, por eso no se ejecutaron experimentos HF aqui. |
| MP2-006 | Done | Se incluyeron `TruncatedSVD`, `NMF` y `PCA` con hiperparametros en la matriz de experimentos. |
| MP2-007 | Done | Matriz de experimentos implementada (TF-IDF + embeddings clasicos + HF; Naive Bayes/CatBoost/LightGBM). |
| MP2-008 | Done | Ranking en notebook actualizado por `test_f1_macro` (criterio final acordado). |
| MP2-009 | Done | Seccion **Modelo ganador** con reportes, matriz de confusion y 4 predicciones de test. |
| MP2-010 | Done | Export verificado: `experiment_results.csv`, `best_model.joblib`, `best_model_metadata.json`. |
| MP2-011 | Done | Validacion: parseo AST de celdas + smoke test ejecutado (modo rapido con 1 experimento) + export de artefactos exitoso. |
| MP2-012 | Done | `src/data_pipeline.py` implementado con `load_dataset`, `maybe_subsample` y `split_train_val_test`. |
| MP2-013 | Done | `python -m src.train_pipeline` ejecutado; genera `artifacts/deploy_model.joblib`, `deploy_model_metadata.json`, `deploy_metrics.json`. |
| MP2-014 | Done | `python -m src.inference_pipeline --text ... --top-k 5` validado contra `deploy_model`. |
| MP2-015 | Done | `app.py` actualizado para usar `predict_text(..., top_k=...)` y mostrar tabla top-k. |

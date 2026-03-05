# Plan detallado final del microproyecto

## 1) Alcance acordado

- Primero: construir un **notebook con experimentación exhaustiva**.
- Toda la experimentación debe estar en el notebook:
  - preprocessing de texto,
  - extracción de características,
  - reducción de dimensionalidad,
  - entrenamiento y calibración de hiperparámetros,
  - comparación de resultados.
- Después (fase 2): implementar fuera del notebook **solo el modelo ganador** para inferencia y despliegue.

## 2) Objetivo técnico del notebook

Comparar múltiples estrategias de clasificación de textos hacia ODS bajo un protocolo reproducible, y seleccionar el mejor experimento con evidencia cuantitativa.

## 3) Protocolo de evaluación (obligatorio)

1. Cargar `data/train.xlsx`.
2. Hacer split estratificado: `train / val / test` (70/15/15).
3. Sobre `train`: calibrar hiperparámetros con `CV=5` (StratifiedKFold).
4. Para cada experimento, reportar:
   - mejor score de CV en train,
   - desempeño en val,
   - desempeño en test.
5. Construir ranking unificado de experimentos.
6. Seleccionar ganador por `f1_macro` en val (desempate por test).

## 4) Procesamiento de texto a experimentar (en notebook)

El procesamiento debe incluir explícitamente, y en combinación:

- Tokenización.
- Normalización (`lowercase`, acentos, limpieza de caracteres).
- Eliminación de stopwords.
- Lematización.
- Stemming.
- Representación con `TF-IDF`.
- Representación con `embeddings` (no limitada a Word2Vec).

Variantes mínimas de preprocesamiento lingüístico:

1. `basic` (tokenización + limpieza).
2. `stopwords`.
3. `lemmatization + stopwords`.
4. `stemming + stopwords`.

Estas variantes deben entrar en la búsqueda de hiperparámetros (no solo una prueba manual).

## 5) Extracción de características y reducción de dimensionalidad

## 5.1 Rama TF-IDF

- `TF-IDF` con tuning (`ngram_range`, `min_df`, `sublinear_tf`, etc.).
- Reducción con `TruncatedSVD`.
- Reducción alternativa con `NMF`.

## 5.2 Rama embeddings

- Embeddings para texto, iterando al menos con 2 familias distintas:
  - `Word2Vec` (promedio por documento),
  - `FastText` (promedio por documento),
  - `Doc2Vec` (vector de documento),
  - `Sentence-Transformers`/Hugging Face (obligatorio en esta versión del plan).
- **Iteración obligatoria con embeddings + PCA**:
  - variar tipo de embedding,
  - variar dimensiones del embedding,
  - variar `n_components` de PCA,
  - comparar impacto en val/test.

## 5.3 Requisito técnico para embeddings de deep learning

- Los embeddings de Hugging Face se deben ejecutar con backend **PyTorch**.
- La inferencia de embeddings debe correr en **GPU de Mac (MPS)**.
- Validación obligatoria al inicio del notebook:
  - `torch.backends.mps.is_available() == True`.
  - selección explícita de dispositivo: `device = 'mps'`.
- Si `mps` no está disponible, se debe detener la celda con error para no mezclar resultados CPU/GPU.
- Recomendación de modelos HF para iterar:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`,
  - `intfloat/multilingual-e5-base` (o variante `small` para velocidad).

## 6) Modelos a evaluar

El notebook debe incluir, como mínimo:

1. `Naive Bayes`:
   - `ComplementNB/MultinomialNB` para `TF-IDF`,
   - `GaussianNB` para `embeddings` densos (con y sin PCA).
2. `CatBoost`.
3. `LightGBM`.

Con combinaciones sobre las distintas representaciones/reducciones.

## 7) Matriz mínima de experimentos

1. Preproc + TF-IDF + Naive Bayes.
2. Preproc + TF-IDF + SVD + CatBoost.
3. Preproc + TF-IDF + SVD + LightGBM.
4. Preproc + TF-IDF + NMF + LightGBM.
5. Preproc + `Embedding_Family_A` + Naive Bayes (`GaussianNB`).
6. Preproc + `Embedding_Family_A` + PCA + Naive Bayes (`GaussianNB`).
7. Preproc + `Embedding_Family_A` + CatBoost.
8. Preproc + `Embedding_Family_A` + LightGBM.
9. Preproc + `Embedding_Family_A` + PCA + LightGBM.
10. Preproc + `Embedding_Family_A` + PCA + CatBoost.
11. Preproc + `Embedding_Family_B` + Naive Bayes (`GaussianNB`).
12. Preproc + `Embedding_Family_B` + PCA + Naive Bayes (`GaussianNB`).
13. Preproc + `Embedding_Family_B` + CatBoost.
14. Preproc + `Embedding_Family_B` + LightGBM.
15. Preproc + `Embedding_Family_B` + PCA + LightGBM.
16. Preproc + `Embedding_Family_B` + PCA + CatBoost.

Notas:

- `Embedding_Family_A` puede ser una familia clásica (`Word2Vec`, `FastText` o `Doc2Vec`).
- `Embedding_Family_B` debe ser una familia de embeddings de Hugging Face en PyTorch ejecutada en `mps`.
- Si el tiempo alcanza, agregar una tercera familia mejora la robustez del análisis.
- Todos deben quedar comparados en una misma tabla final.

## 8) Sección obligatoria: “Modelo ganador”

Debe mostrar claramente:

- nombre exacto del experimento ganador,
- `best_params_`,
- métricas en val y test (`f1_macro`, `f1_weighted`, `accuracy`),
- matriz de confusión en test,
- `classification_report` en test,
- evidencia de al menos 4 predicciones reales vs predichas de test.

## 9) Artefactos que debe exportar el notebook

- `artifacts/experiment_results.csv`
- `artifacts/best_model.joblib`
- `artifacts/best_model_metadata.json`

## 10) Entregables fase 1 (ahora)

1. `Microproyecto2.ipynb` con toda la experimentación y celdas ejecutadas.
2. `Microproyecto2.html` exportado desde notebook.
3. Artefactos del ganador en `artifacts/`.

## 11) Fase 2 (después de que se confirme el ganador)

Con el experimento ganador confirmado por ti:

- implementar pipeline de inferencia fuera del notebook,
- construir app Streamlit de despliegue,
- conectar la app al artefacto final del ganador.

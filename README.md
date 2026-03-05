# ODS Prediction - Microproyecto 2

Proyecto de clasificacion de textos para predecir el Objetivo de Desarrollo Sostenible (ODS) asociado a un texto en espanol.

Este trabajo forma parte del curso **Aprendizaje No Supervisado** de la **Maestria en Inteligencia Artificial** (Universidad de los Andes, Colombia).

## 1. Alcance

El repositorio incluye:
- Notebook de experimentacion exhaustiva: preprocesamiento, TF-IDF, reduccion de dimensionalidad y embeddings.
- Seleccion del modelo ganador por **F1 macro en test**.
- Implementacion productiva fuera del notebook:
  - pipeline de datos,
  - pipeline de entrenamiento,
  - pipeline de inferencia,
  - aplicacion Streamlit.

## 2. Estructura del repositorio

```text
.
|-- Microproyecto2.ipynb
|-- app.py
|-- data/
|   `-- train.xlsx
|-- artifacts/
|   |-- best_model.joblib
|   |-- best_model_metadata.json
|   |-- deploy_model.joblib
|   |-- deploy_model_metadata.json
|   `-- deploy_metrics.json
|-- assets/
|   |-- uniandes_logo.svg
|   `-- uniandes_wordmark.svg
`-- src/
    |-- config.py
    |-- data_pipeline.py
    |-- text_features.py
    |-- train_pipeline.py
    `-- inference_pipeline.py
```

## 3. Datos esperados

Archivo: `data/train.xlsx`

Columnas obligatorias:
- `textos`: texto en espanol
- `ODS`: etiqueta objetivo (entero)

## 4. Requisitos

Recomendado:
- Python 3.11
- pip actualizado

Dependencias principales:
- streamlit
- pandas
- numpy
- scikit-learn
- nltk
- spacy
- gensim
- sentence-transformers
- torch
- catboost
- lightgbm
- joblib
- matplotlib
- openpyxl

Instalacion sugerida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para despliegue en Streamlit Community Cloud, este repositorio incluye `requirements.txt`.

## 5. Notebook de experimentacion

El notebook principal es:

```bash
jupyter notebook Microproyecto2.ipynb
```

Notas:
- El notebook es autocontenido para sus transformadores (no depende de `src/` para esas clases).
- Descargas iniciales de NLTK/spaCy/HuggingFace pueden tardar la primera vez.

## 6. Entrenamiento productivo (fuera del notebook)

Entrena y guarda el modelo de despliegue en `artifacts/deploy_model.joblib` usando el ganador del notebook:

```bash
python -m src.train_pipeline
```

Opciones utiles:

```bash
python -m src.train_pipeline --hf-device auto
python -m src.train_pipeline --sample-size 3000
python -m src.train_pipeline --allow-notebook-fallback
```

Salidas:
- `artifacts/deploy_model.joblib`
- `artifacts/deploy_model_metadata.json`
- `artifacts/deploy_metrics.json`

## 7. Inferencia por linea de comandos

```bash
python -m src.inference_pipeline --text "Programa de reciclaje y economia circular en barrios" --top-k 1
```

## 8. Aplicacion Streamlit

Ejecutar localmente:

```bash
streamlit run app.py
```

Caracteristicas de la app:
- UI minimalista con branding Uniandes.
- Prediccion ODS y confianza del modelo.
- `top_k` fijo en 1 (sin control editable en interfaz).

## 9. Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub.
2. En Streamlit Cloud, crear app desde el repo.
3. Definir `app.py` como entrypoint.
4. Usar Python 3.11 en configuracion avanzada.
5. Deploy.

## 10. Criterio principal de seleccion

La comparacion de experimentos se realiza con metrica principal:
- **test_f1_macro**

El modelo ganador y sus hiperparametros quedan registrados en metadata dentro de `artifacts/`.

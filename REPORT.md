# Automated Customer Review Analysis — Project Report
**IronHack AI Engineering Bootcamp · 2025**
**Author:** Francisco
**GitHub:** https://github.com/franromlob/Project_Automated_Customer_Reviews

---

## Links del Proyecto (para la presentación)

| Recurso | URL |
|---|---|
| 🌐 Web App (Streamlit) | https://projectautomatedcustomerreviews-yjfrjpfxn2s3fnyqdqqrlk.streamlit.app/ |
| 🤗 Modelo en HuggingFace Hub | https://huggingface.co/FranMRL/roberta-amazon-sentiment |
| 🎨 Demo Gradio (Space) | https://huggingface.co/spaces/FranMRL/amazon-review-sentiment |

---

## Estructura de la Presentación (13 slides · ~15 min)

---

### SLIDE 1 — Portada

**Título:** Automated Customer Review Analysis
**Subtítulo:** From Raw Reviews to Product Recommendations with NLP & Generative AI
**Bootcamp:** IronHack AI Engineering · 2025

---

### SLIDE 2 — El Problema

**¿Por qué automatizar el análisis de reseñas?**

- Amazon genera millones de reseñas diarias — inabordables manualmente
- Los equipos de producto necesitan feedback rápido y estructurado
- Las reseñas contienen señales valiosas: calidad percibida, defectos, perfil del comprador

**Objetivo del proyecto:**
Construir un pipeline end-to-end que transforme reseñas en lenguaje natural en artículos de recomendación listos para publicar.

---

### SLIDE 3 — Arquitectura del Sistema

```
Raw Reviews (CSV)
      ↓
  01 · Preprocessing     ← pandas · limpieza · deduplicación · etiquetado
      ↓
  02 · Classification    ← RoBERTa fine-tuned → positive / neutral / negative
      ↓
  03 · Clustering        ← TF-IDF + PCA + K-Means → grupos temáticos
      ↓
  04 · Summarization     ← GPT-4o-mini + Prompt Engineering → artículos
      ↓
  05 · Deployment        ← Streamlit App + HuggingFace Hub + Gradio Space
```

**Puntos clave a destacar:**
- Cada módulo es independiente (`src/preprocess.py`, `src/classify.py`, `src/cluster.py`, `src/summarize.py`)
- El pipeline puede ejecutarse sobre cualquier CSV de reseñas (Tab "Upload & Analyse")

---

### SLIDE 4 — Dataset & Preprocesamiento *(15 pts)*

**Dataset:** Amazon Product Reviews

| Métrica | Valor |
|---|---|
| Total reseñas (limpias) | 721 |
| Productos únicos | 23 |
| Categorías | 2 (Electronics, Office Supplies) |
| Rating medio | 4.64 / 5.0 |
| Distribución sentimiento | 93.8% positivo · 4% neutral · 2.2% negativo |

**Pipeline de preprocesamiento (`src/preprocess.py`):**

| Paso | Operación | Función |
|---|---|---|
| Texto | Lowercase + strip de espacios | `clean_text()` |
| Tipos | `pd.to_numeric(errors="coerce")` para ratings | `add_sentiment_column()` |
| Etiquetado | Rating 1-2 → negative · 3 → neutral · 4-5 → positive | `map_sentiment()` |
| Deduplicación | Eliminación por texto exacto | `remove_duplicates()` |

**Gráficos a insertar en este slide:**
- `data/processed/rating_distribution.png` — distribución de ratings
- `data/processed/sentiment_distribution.png` — distribución de sentimientos
- `data/processed/class_imbalance.png` — desbalance de clases (justifica el class-weighting)

---

### SLIDE 5 — EDA: Insights del Dataset

**Gráficos a insertar:**
- `data/processed/top_products.png` — productos con más reseñas
- `data/processed/avg_rating_by_product.png` — rating medio por producto
- `data/processed/review_lengths.png` — longitud de las reseñas (justifica MAX_LENGTH=128)
- `data/processed/rating_variability.png` — variabilidad por producto

**Insight clave para la audiencia no técnica:**
> "El 93.8% de las reseñas son positivas — esto crea un desbalance de clases severo que el modelo tiene que aprender a manejar."

---

### SLIDE 6 — Modelo 1: Clasificación de Sentimiento *(20 pts)*

**Comparativa de 4 modelos:**

| Modelo | Tipo | Accuracy | F1 (weighted) |
|---|---|---|---|
| VADER | Rule-based (baseline) | ~0.65 | ~0.61 |
| nlptown BERT | Pretrained | ~0.72 | ~0.70 |
| RoBERTa base | Pretrained | ~0.78 | ~0.76 |
| **RoBERTa fine-tuned** | **Fine-tuned** | **~0.84** | **~0.82** |

**Gráfico a insertar:**
- `data/processed/model_comparison_4models.png` — comparativa visual de los 4 modelos

**¿Por qué RoBERTa?**
- Arquitectura Transformer bidireccional más moderna que BERT
- Pre-entrenado en ~160GB de texto (tweets, web, libros)
- El fine-tuning adapta el dominio: de tweets → reseñas de Amazon

---

### SLIDE 7 — Fine-Tuning: Técnica y Resultados

**Reto técnico:** Clase "neutral" muy minoritaria (4%) → el modelo ignoraría esa clase

**Solución: Class-Weighted Loss**
```python
# Penaliza más los errores en clases minoritarias
class_weights = compute_class_weight("balanced", classes=["neg","neu","pos"], y=y_train)
loss_fn = CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
```

**Resultado del fine-tuning:**
- +6 puntos de F1 respecto al modelo base RoBERTa
- Mejora especialmente en la clase "neutral" (la más difícil)

**Gráficos a insertar:**
- `data/processed/confusion_matrix_roberta.png` — modelo base (antes del fine-tuning)
- `data/processed/confusion_matrix_finetuned.png` — modelo fine-tuned (después)

**Diferencia visual:** la diagonal principal más marcada = menos errores de clasificación

---

### SLIDE 8 — Modelo 2: Clustering *(20 pts)*

**Objetivo:** descubrir grupos temáticos dentro de las reseñas sin etiquetas previas (aprendizaje no supervisado)

**Pipeline (`src/cluster.py`):**

```
Texto de reseña
      ↓
TF-IDF Vectorizer (1.000 features)   ← representación numérica del texto
      ↓
PCA (50 componentes)                  ← reducción de dimensionalidad
      ↓
K-Means                               ← asignación a clusters
      ↓
Silhouette Score → selección K óptimo
```

**¿Por qué K-Means sobre otras alternativas?**

| Algoritmo | Escalabilidad | Interpretabilidad | Elección |
|---|---|---|---|
| K-Means | O(n) — rápido | Alta (centroides) | ✅ Elegido |
| DBSCAN | O(n²) — lento | Media | ❌ Descartado |
| Jerárquico | O(n²) — muy lento | Alta | ❌ Descartado |
| GMM | O(n·k) — moderado | Media | ❌ Descartado |

**Gráficos a insertar:**
- `data/processed/clustering_optimal_k.png` — Silhouette Score por valor de K
- `data/processed/clustering_visualization.png` — clusters en espacio PCA 2D
- `data/processed/cluster_statistics.png` — tamaño y distribución de clusters

---

### SLIDE 9 — Modelo 3: Summarización y Artículos *(20 pts)*

**Objetivo:** generar artículos de recomendación accionables a partir de las reseñas clasificadas

**Arquitectura del prompt (GPT-4o-mini):**
```
System: "You are a professional product analyst..."

User: "Based on these {n} reviews for {product}:
       [review_1, review_2, ..., review_n]

       Generate an article with:
       1. Executive Summary
       2. Key Strengths (from positive reviews)
       3. Areas for Improvement (from negative/neutral)
       4. Ideal Buyer Profile
       5. Final Recommendation with score X/5"
```

**Parámetros:** `temperature=0.7` · `max_tokens=1000` · modelo `gpt-4o-mini`

**Resultado:**
- 4 artículos generados (uno por categoría de producto)
- Disponibles en `data/processed/articles/`
- Categorías: Alexa Devices · Fire Tablets · E-Readers & Accessories · E-Readers Premium

**Ejemplo de salida (Alexa Devices):**
> *"The Echo Plus is celebrated for its seamless integration with smart home devices [...] ideal for users looking to create an extensive smart home ecosystem"*

---

### SLIDE 10 — Resultado Final: Artículo Generado

**Insertar fragmento del artículo más completo** (por ejemplo, `alexa_devices.txt`):

Estructura del artículo:
- **Top 3 Products** con puntuaciones y diferenciadores clave
- **Top Complaints** por producto con ejemplos reales
- **Product to Avoid** con justificación basada en datos
- **Recommended Product** con perfil de comprador ideal

> Este es el entregable final del pipeline — un artículo listo para publicar en un blog de e-commerce, generado automáticamente a partir de las reseñas de clientes.

---

### SLIDE 11 — Despliegue *(10 pts + 10 pts extra)*

**Tres componentes desplegados y públicos:**

| Componente | Tecnología | URL |
|---|---|---|
| 🌐 Web App completa | Streamlit Community Cloud | https://projectautomatedcustomerreviews-yjfrjpfxn2s3fnyqdqqrlk.streamlit.app/ |
| 🤗 Modelo en la nube | HuggingFace Hub | https://huggingface.co/FranMRL/roberta-amazon-sentiment |
| 🎨 Demo interactivo | HuggingFace Spaces + Gradio | https://huggingface.co/spaces/FranMRL/amazon-review-sentiment |

**La web app tiene 4 funcionalidades:**
1. **Sentiment Classifier** — clasifica cualquier reseña en tiempo real con % de confianza
2. **Product Categories** — explora los 23 productos (donut chart + métricas + hashtags automáticos)
3. **Recommendation Articles** — artículos GPT listos para publicar por categoría
4. **Upload & Analyse** — sube tu propio CSV y el pipeline completo corre automáticamente

**Insertar screenshot de la web app aquí**

---

### SLIDE 12 — Stack Tecnológico

| Área | Herramientas |
|---|---|
| **Data** | pandas, numpy |
| **NLP / ML** | scikit-learn, HuggingFace Transformers, PyTorch |
| **GenAI** | OpenAI GPT-4o-mini, Prompt Engineering |
| **Visualización** | Plotly, Streamlit |
| **Despliegue** | Streamlit Community Cloud, HuggingFace Hub, Gradio |
| **Calidad de código** | pylint, PEP 8, type hints, Google-style docstrings |
| **Seguridad** | python-dotenv, .env, secrets.toml (sin hardcode de API keys) |

---

### SLIDE 13 — Conclusiones y Próximos Pasos

**Resultados obtenidos:**

| Métrica | Valor |
|---|---|
| Accuracy clasificador (fine-tuned) | ~84% |
| F1-Score weighted | ~82% |
| Mejora sobre baseline (VADER) | +19 puntos F1 |
| Productos analizados | 23 |
| Artículos generados | 4 categorías |
| App pública accesible | ✅ |

**Lo que funciona bien:**
- El fine-tuning de RoBERTa mejora significativamente la clasificación en e-commerce
- La selección automática de K hace el clustering robusto sin intervención manual
- GPT-4o-mini con prompt engineering genera artículos coherentes y accionables

**Próximos pasos:**
- Conectar la Streamlit app al modelo en HuggingFace Hub (inferencia 100% en la nube)
- Soporte multilingüe para reseñas en español, alemán, francés
- Dashboard de monitorización de drift del modelo en producción (MLOps)

---

## Guía de Gráficos por Slide

| Slide | Archivo | Descripción |
|---|---|---|
| 4 | `data/processed/rating_distribution.png` | Distribución de ratings 1-5 |
| 4 | `data/processed/sentiment_distribution.png` | % positivo / neutral / negativo |
| 4 | `data/processed/class_imbalance.png` | Desbalance de clases (justifica class-weighting) |
| 5 | `data/processed/top_products.png` | Top productos por número de reseñas |
| 5 | `data/processed/avg_rating_by_product.png` | Rating medio por producto |
| 5 | `data/processed/review_lengths.png` | Longitud de reseñas (justifica MAX_LENGTH=128) |
| 6 | `data/processed/model_comparison_4models.png` | Comparativa accuracy/F1 de los 4 modelos |
| 7 | `data/processed/confusion_matrix_roberta.png` | Matriz de confusión — modelo base |
| 7 | `data/processed/confusion_matrix_finetuned.png` | Matriz de confusión — fine-tuned |
| 8 | `data/processed/clustering_optimal_k.png` | Silhouette Score para selección de K |
| 8 | `data/processed/clustering_visualization.png` | Clusters en espacio 2D (PCA) |
| 8 | `data/processed/cluster_statistics.png` | Distribución de tamaño de clusters |

---

## Notas para la Presentación

### Audiencia técnica (profesor, evaluadores)
- Enfatizar: class-weighted loss, Silhouette Score para K óptimo, arquitectura Transformer
- Mostrar: confusion matrices antes/después del fine-tuning
- Hablar de: trade-offs entre modelos (K-Means vs DBSCAN), temperatura en GPT

### Audiencia no técnica
- Enfatizar: "el sistema lee 721 reseñas y genera un artículo listo para publicar en segundos"
- Mostrar: la web app en vivo con una demo en directo
- Evitar: fórmulas matemáticas, hablar de arquitecturas internas

### Timing sugerido (15 min)
- Slides 1-3: 2 min (contexto y arquitectura)
- Slides 4-5: 2 min (datos)
- Slides 6-7: 3 min (clasificación + fine-tuning)
- Slide 8: 2 min (clustering)
- Slide 9-10: 2 min (summarización + ejemplo)
- Slides 11-13: 4 min (demo en vivo + conclusiones)

---

## Criterios de Evaluación — Checklist

| Criterio | Puntos | Estado |
|---|---|---|
| Preprocesamiento de datos | 15 | ✅ `src/preprocess.py` + `01_eda.ipynb` |
| Modelo de clasificación | 20 | ✅ `src/classify.py` + `02_classification.ipynb` |
| Modelo de clustering | 20 | ✅ `src/cluster.py` + `03_clustering.ipynb` |
| Modelo de summarización | 20 | ✅ `src/summarize.py` + `04_summarization.ipynb` |
| Despliegue del modelo | 10 | ✅ Streamlit Community Cloud |
| Informe en PDF | 5 | ✅ Este documento (exportar a PDF) |
| Presentación PPT | 10 | ✅ Basada en este informe |
| Extra: alojamiento público | 10 | ✅ HuggingFace Hub + Gradio Space |
| **TOTAL** | **110** | |

**Puntuación mínima para aprobar: 70 puntos**

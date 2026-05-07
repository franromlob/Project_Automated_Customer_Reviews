# Guía de Estudio — Explicación del Código al Profesor

> **Cómo usar este documento**  
> Lee cada sección antes de la presentación. La estructura sigue el orden real de ejecución del pipeline.  
> Cada módulo incluye: qué hace, decisiones clave de diseño, y preguntas frecuentes con respuesta lista.

---

## 0. El Pipeline Completo — Visión General

```
datos raw (CSV)
      │
      ▼
┌─────────────────┐
│  preprocess.py  │  limpia, normaliza y añade etiqueta de sentimiento
└────────┬────────┘
         │  reviews_clean.csv
         ▼
┌─────────────────┐
│   classify.py   │  RoBERTa predice sentimiento por reseña
└────────┬────────┘
         │  reviews_with_predictions_finetuned.csv
         ▼
┌─────────────────┐
│   cluster.py    │  TF-IDF + K-Means agrupa productos en 5 categorías
└────────┬────────┘
         │  reviews_clustered.csv
         ▼
┌─────────────────┐
│  summarize.py   │  GPT-4o-mini genera artículo por categoría
└────────┬────────┘
         │  articles_summary.csv
         ▼
┌─────────────────┐
│   app/main.py   │  Streamlit integra los 3 módulos en una web app
└─────────────────┘
```

**Mensaje clave para el profesor:**  
> "Cada módulo es independiente, con entrada y salida en CSV. Esto permite sustituir cualquier componente sin romper el resto — principio de bajo acoplamiento."

---

## 1. `src/preprocess.py` — El Pipeline de Datos

### Qué hace
Transforma el CSV raw de Amazon (5.000 filas, muchas columnas basura) en un dataset limpio con 4.805 reseñas listas para el modelo.

### Ejecución
```bash
python src/preprocess.py
# Input:  data/raw/amazon_reviews.csv
# Output: data/processed/reviews_clean.csv
```

### Arquitectura del módulo
El módulo usa el patrón **functional pipeline**: cada función recibe un DataFrame y devuelve un DataFrame modificado. La función `main()` los encadena en orden.

```python
def main():
    df = load_data(RAW_DATA_PATH)       # línea 149
    df = select_columns(df, ...)        # línea 150
    df = rename_columns(df)             # línea 151
    df = drop_nulls(df)                 # línea 152
    df = drop_duplicates(df)            # línea 153
    df = clean_text(df)                 # línea 154
    df = clean_categories(df)           # línea 155
    df = add_sentiment_column(df)       # línea 156
    save_data(df, PROCESSED_DATA_PATH)  # línea 157
```

### Funciones clave — qué decirle al profesor

**`map_sentiment(rating)` — línea 96**
```python
def map_sentiment(rating: float) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"
```
> "Esta función es el punto de entrada de las etiquetas de supervisión. Transformo una variable continua (1-5 estrellas) en una categórica de 3 clases. Es lo que permite usar el dataset como *ground truth* para clasificación supervisada."

**`clean_text(df)` — línea 84**
```python
df["review_text"] = df["review_text"].astype(str).str.strip().str.lower()
```
> "Normalizo el texto en minúsculas para que el modelo no trate 'Good' y 'good' como palabras distintas. El tokenizador de RoBERTa es case-sensitive en su versión base, aunque el nuestro usa la variante uncased."

**`drop_duplicates` — línea 75**
```python
df = df.drop_duplicates(subset=["review_text", "product_name"])
```
> "Elimino duplicados sobre el par texto+producto, no solo sobre el texto. Una misma reseña puede aparecer en dos productos distintos y ser válida."

### Posibles preguntas del profesor

**P: ¿Por qué no usas SMOTE u oversampling aquí?**  
R: "El desbalanceo (94% positive) es real — refleja el comportamiento de los usuarios de Amazon. SMOTE generaría reseñas sintéticas que no existirían en producción. Decido preservar la distribución real y gestionar el desbalanceo en el modelo con *class weights*, que es más honesto con los datos."

**P: ¿Qué pasaría si una reseña tiene 3.5 estrellas?**  
R: "Con `pd.to_numeric(errors='coerce')` convierto el rating a float. Un 3.5 entraría en el `return 'positive'` del `map_sentiment` porque solo el 3.0 exacto mapea a neutral. En el dataset real los ratings son enteros, pero el código lo gestiona."

---

## 2. `src/classify.py` — Clasificación con Transformers

### Qué hace
Carga RoBERTa (pretrained o fine-tuned), tokeniza las reseñas en batches de 32 y devuelve una etiqueta de sentimiento + score de confianza para cada reseña.

### Concepto central a explicar: el pipeline de inferencia

```
texto plano
     │
     ▼  tokenizer(text, truncation=True, max_length=128, padding="max_length")
token IDs + attention mask (tensores)
     │
     ▼  model(**encoding)
logits  [batch_size × 3]   ← puntuaciones crudas, sin normalizar
     │
     ▼  torch.argmax(logits, dim=1)
índice de la clase ganadora (0=positive, 1=neutral, 2=negative)
     │
     ▼  id2label[pred]
"positive" / "neutral" / "negative"
```

### Funciones clave — qué decirle al profesor

**`load_model(model_path)` — línea 27**
```python
def load_model(model_path: str = None):
    if model_path and os.path.isdir(model_path):
        source = model_path          # fine-tuned local
    else:
        source = MODEL_NAME          # Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    id2label = model.config.id2label
    model = model.to(device)
    model.eval()                     # ← desactiva dropout
```
> "Uso `Auto` classes de HuggingFace para que el código funcione con cualquier arquitectura Transformer sin cambiar nada. El `model.eval()` es crítico: desactiva el *dropout* y el *batch normalization* en modo inferencia — sin esto, el modelo daría resultados distintos en cada predicción."

**`predict_sentiment` — línea 66 (batch inference)**
```python
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    encoding = tokenizer(batch, truncation=True, ...)
    with torch.no_grad():            # ← no construye el grafo de gradientes
        outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=1)
```
> "`torch.no_grad()` es la clave de la eficiencia en inferencia. Durante el entrenamiento necesitamos el grafo para el *backpropagation*, pero en inferencia es pure overhead. Este context manager lo desactiva y reduce el uso de memoria ~50%."

**`predict_single` — línea 114 (confianza con softmax)**
```python
probs = torch.softmax(outputs.logits, dim=1)
pred = torch.argmax(probs, dim=1).item()
confidence = probs[0][pred].item()
```
> "Los `logits` son puntuaciones sin normalizar — pueden ser cualquier número real. `softmax` los convierte en probabilidades que suman 1. La confianza es simplemente la probabilidad de la clase predicha. Si el modelo devuelve [0.95, 0.03, 0.02], la confianza es 95%."

### Por qué 128 tokens y no 512
> "La longitud máxima de BERT/RoBERTa es 512 tokens. Nuestra mediana de reseñas es 21 palabras ≈ 25 tokens. El percentil 95 es 80 palabras ≈ 96 tokens. Usar 128 cubre el 95% de los casos con un buffer, y es 4× más rápido que usar 512 porque la atención cuadrática de los Transformers escala como O(n²) respecto a la longitud."

### Posibles preguntas del profesor

**P: ¿Qué son los logits?**  
R: "Son las activaciones de la capa final del modelo antes de aplicar ninguna función de activación. En clasificación con CrossEntropyLoss, PyTorch espera logits crudos — la función de pérdida aplica softmax internamente. Por eso la capa final de RoBERTa no tiene softmax."

**P: ¿Qué diferencia hay entre `predict_sentiment` y `predict_single`?**  
R: "La diferencia principal es el `softmax`. Para clasificación en batch solo necesitamos la etiqueta (argmax), es más rápido. Para la web app necesitamos también la confianza para mostrarla en el gauge, así que aplicamos softmax y extraemos la probabilidad de la clase ganadora."

**P: ¿Qué hace `model.config.id2label`?**  
R: "HuggingFace guarda la correspondencia entre índices numéricos y etiquetas dentro de la configuración del modelo. Al cargar el modelo fine-tuned que guardamos con `model.save_pretrained()`, este diccionario se restaura automáticamente. No hay que hardcodear nada."

---

## 3. `src/cluster.py` — Clustering Unsupervised

### Qué hace
Agrupa los 23 productos únicos en 5 meta-categorías usando TF-IDF + PCA + K-Means, sin ninguna etiqueta supervisada.

### El pipeline completo

```
reviews por producto (texto concatenado)
        │
        ▼  TfidfVectorizer(max_features=500, ngram_range=(1,2))
matriz TF-IDF  [23 productos × 500 features]
        │
        ▼  PCA(n_components=10)    → 89.15% varianza explicada
coordenadas reducidas  [23 × 10]
        │
        ▼  KMeans(n_clusters=K)   evaluado para K=2..8
labels de cluster  [23 valores]
        │
        ▼  map cluster_id → nombre legible
meta_category por producto
```

### Funciones clave — qué decirle al profesor

**`aggregate_reviews` — línea 34**
```python
product_df["combined_text"] = (
    product_df["product_name"] + " " +
    product_df["product_name"] + " " +    # ← duplicado a propósito
    product_df["all_reviews"]
)
```
> "Duplicar el nombre del producto en el texto combinado es un truco clásico de TF-IDF *boosting*. El nombre del producto contiene las palabras más discriminativas ('Fire', 'Kindle', 'Echo') y quiero que tengan más peso relativo frente al ruido de miles de reseñas."

**`build_tfidf_matrix` — línea 67**
```python
tfidf = TfidfVectorizer(
    max_features=500,     # las 500 palabras más informativas
    stop_words="english", # elimina 'the', 'a', 'is', etc.
    ngram_range=(1, 2),   # unigramas + bigramas ('kindle fire', 'battery life')
    min_df=1
)
```
> "TF-IDF pondera cada palabra por su frecuencia en el documento versus su rareza en el corpus. Una palabra que aparece en todos los productos (como 'Amazon') tendrá IDF bajo y poco peso. Una palabra que aparece solo en los Kindle (como 'e-ink') tendrá IDF alto y será discriminativa para el clustering."

**`find_optimal_k` — línea 105**
```python
for k in k_range:
    kmeans = KMeans(n_clusters=k, ...)
    labels = kmeans.fit_predict(coords)
    sil = silhouette_score(coords, labels)
```
> "Evalúo dos métricas: inercia (elbow method) y Silhouette score. La inercia sola no es suficiente porque siempre decrece al aumentar K. El Silhouette mide cohesión interna versus separación entre clusters, con rango [-1, 1]. Nuestro K=5 tiene score 0.40 — moderado, esperado con solo 23 productos y vocabulario compartido entre categorías."

**`reduce_dimensions` — línea 88 — por qué PCA antes de K-Means**
> "La matriz TF-IDF tiene 500 dimensiones. K-Means usa distancia euclidiana, que en alta dimensionalidad sufre la *curse of dimensionality* — todas las distancias tienden a ser similares y los clusters pierden significado. PCA a 10 componentes conserva el 89.15% de la varianza y hace K-Means viable. Esta combinación TF-IDF + PCA + K-Means es el pipeline estándar de clustering de texto antes de BERT."

### Posibles preguntas del profesor

**P: ¿Por qué K-Means y no DBSCAN o clustering jerárquico?**  
R: "Con 23 productos K-Means es suficiente. DBSCAN requiere definir `epsilon` y funciona bien con clusters de forma arbitraria y muchos puntos, pero con 23 puntos en 10 dimensiones es frágil. El clustering jerárquico podría ser interesante para ver el dendrograma, pero al final el objetivo es asignar cada producto a exactamente una categoría, que es lo que hace K-Means."

**P: ¿Qué significa Silhouette score = 0.40?**  
R: "El Silhouette va de -1 a +1. 1.0 = clusters perfectamente separados. 0.0 = clusters solapados. 0.40 es moderado — los clusters son coherentes pero no perfectamente separados. Es esperable: todos los productos son de Amazon, comparten vocabulario ('battery', 'screen', 'app') y solo se diferencian en términos específicos como 'kindle', 'fire', 'echo'."

**P: ¿Cómo asignas nombres a los clusters?**  
R: "Manualmente, analizando qué productos cayeron en cada cluster. K-Means no genera nombres, solo números. Después de ver que el cluster 1 agrupaba Fire HD 8, Fire HD 10, etc., le asigné 'Fire Tablets'. Esto es el paso de interpretación — inherentemente humano en clustering."

---

## 4. `src/summarize.py` — Generación con GPT-4o-mini

### Qué hace
Para cada meta-categoría, construye un prompt estructurado con los mejores y peores productos (según reviews reales) y llama a GPT-4o-mini para generar un artículo de recomendación de 400-600 palabras.

### El pipeline

```
reviews_clustered.csv
        │
        ▼  prepare_category_data(df, category)
dict con: stats por producto, top 5 reviews positivas/negativas por producto
        │
        ▼  build_prompt(category_data)
string con el prompt completo (instrucciones + datos de reviews)
        │
        ▼  generate_article(client, prompt, category)
texto del artículo generado por GPT
        │
        ▼  save_articles(articles, output_dir)
archivos .txt + articles_summary.csv
```

### Funciones clave — qué decirle al profesor

**`get_client()` — línea 29 — seguridad de credenciales**
```python
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in the .env file.")
    return OpenAI(api_key=api_key)
```
> "La API key nunca está hardcodeada — se lee de la variable de entorno `OPENAI_API_KEY` que vive en `.env`, excluido del repositorio por `.gitignore`. Si alguien clona el repo sin el `.env`, el código falla con un mensaje claro en lugar de enviar una key vacía a la API."

**`prepare_category_data` — línea 43 — selección de reviews**
```python
positive = prod_df[prod_df["predicted_sentiment_finetuned"] == "positive"]["review_text"].head(5)
negative = prod_df[prod_df["predicted_sentiment_finetuned"] == "negative"]["review_text"].head(5)
```
> "Las reviews que le paso a GPT están filtradas por la predicción del modelo RoBERTa — no por el rating. Esto crea un bucle: el clasificador de sentimiento alimenta al generador de artículos. Si el clasificador se equivoca, el artículo puede recibir ejemplos incorrectos. Es la dependencia más importante entre módulos del pipeline."

**`build_prompt` — línea 105 — prompt engineering**

La estructura del prompt tiene tres partes:
1. **System context**: "You are a tech product reviewer writing for a consumer advice website"
2. **Data injection**: producto, rating, muestras de reviews positivas/negativas
3. **Output specification**: 5 secciones obligatorias con nombres explícitos

> "El prompt engineering aquí sigue el patrón *role + data + format*. Definir un rol específico ('consumer advice website') ancla el tono. Inyectar datos reales evita alucinaciones — GPT cita reviews que existen. Especificar las 5 secciones con nombres exactos garantiza estructura consistente entre artículos."

**`generate_article` — línea 168 — parámetros de generación**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user",   "content": prompt}
    ],
    max_tokens=1000,
    temperature=0.7
)
```
> "`temperature=0.7` es el balance entre creatividad y repetibilidad. Temperature=0 hace el modelo determinista (siempre el mismo output). Temperature=1 hace el output más variado pero menos controlado. 0.7 es el estándar para generación de texto informativo. `max_tokens=1000` protege contra outputs excesivamente largos que dispararían el coste de la API."

### Posibles preguntas del profesor

**P: ¿Por qué GPT-4o-mini y no T5 o BART?**  
R: "T5 y BART son modelos extractivos/abstractivos entrenados para resumir un solo documento. Aquí el problema es distinto: tengo múltiples productos con múltiples reseñas y necesito generar un artículo con estructura personalizada. GPT-4o-mini sigue instrucciones complejas sin fine-tuning, lo que hace innecesario construir un dataset de entrenamiento. El coste es ínfimo (~$0.01 por artículo) para este volumen."

**P: ¿Cómo evalúas la calidad del artículo?**  
R: "No tengo evaluación automática — no hay referencias escritas por humanos para calcular ROUGE o BLEU. La evaluación es cualitativa: verifico que las 5 secciones estén presentes, que cite productos reales del dataset y que el tono sea apropiado. Esto es una limitación real del proyecto que reconozco en las conclusiones."

**P: ¿Qué pasa si GPT no devuelve la estructura correcta?**  
R: "En los 4 artículos generados el modelo siguió la estructura. Con temperature=0.7 y secciones nombradas explícitamente la tasa de conformidad es muy alta. Para producción añadiría un parser que verifique la presencia de cada sección y reintente si falta alguna — eso sería el siguiente paso."

---

## 5. `app/main.py` — La Web App con Streamlit

### Qué hace
Integra los tres módulos en una interfaz de 3 tabs accesible para usuarios no técnicos. Carga el modelo una sola vez en memoria y gestiona el estado entre interacciones.

### Decisiones de diseño clave

**`@st.cache_data` vs `@st.cache_resource` — líneas 62-77**
```python
@st.cache_data
def load_data() -> pd.DataFrame:      # datos → cacheados como objeto serializable
    return pd.read_csv(...)

@st.cache_resource
def load_classifier():                 # modelo ML → cacheado como recurso compartido
    return load_model(...)
```
> "Esta diferencia es importante. `@st.cache_data` es para datos serializables (DataFrames, listas). `@st.cache_resource` es para objetos que no se pueden serializar — como un modelo PyTorch en GPU. Sin este caché, Streamlit recargaría RoBERTa (~500MB) en cada interacción del usuario."

**`sys.path.insert(0, os.getcwd())` — línea 17**
```python
sys.path.insert(0, os.getcwd())
from src.classify import load_model, predict_single
```
> "Streamlit ejecuta `app/main.py` desde cualquier directorio. Esta línea añade la raíz del proyecto al path de Python para que los imports `from src.classify` funcionen independientemente de desde dónde se lance. Es una solución de compatibilidad, no el patrón ideal — en producción usaríamos un paquete instalable con `pip install -e .`"

**El gauge chart de confianza — línea 155**
```python
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(confidence * 100, 1),
    gauge={"bar": {"color": SENTIMENT_COLORS[sentiment]}, ...}
))
```
> "El color del gauge es dinámico — cambia según el sentimiento predicho (verde/amarillo/rojo). Esto da feedback visual inmediato al usuario sin necesidad de leer el texto. Pequeño detalle de UX que marca la diferencia en una demo."

**Manejo de errores en Tab 1 — línea 180**
```python
except (OSError, RuntimeError, ValueError) as e:
    st.error(f"Model error: {e}")
    st.info("Make sure the fine-tuned model exists in `models/roberta_finetuned`...")
```
> "Capturo errores específicos, no un `except Exception` genérico. Si el modelo fine-tuned no está descargado, el error es descriptivo y dice exactamente qué hacer. Esto cumple el principio de *fail loudly with context* — mejor un error claro que un crash silencioso."

### Posibles preguntas del profesor

**P: ¿Por qué Streamlit y no Flask o FastAPI?**  
R: "Para un prototipo orientado a una audiencia de marketing/producto, Streamlit es la elección correcta: cero HTML/CSS/JavaScript, interactividad con widgets nativos de Python, y despliegue trivial. Flask o FastAPI tendrían más sentido si necesitara una API REST consumida por un frontend separado o si el performance bajo carga fuera crítico."

**P: ¿Cómo escalarías esto a producción?**  
R: "Tres cambios principales: 1) Hospedar el modelo en HuggingFace Spaces con GPU compartida, eliminando la dependencia local. 2) Mover la API key de OpenAI a Streamlit Secrets en lugar de `.env`. 3) Añadir un endpoint REST separado para la clasificación de sentimiento para que otros servicios puedan consumirlo, desacoplando la inferencia de la UI."

---

## 6. Preguntas Generales sobre el Proyecto

**P: ¿Cuál fue el mayor reto técnico?**  
R: "El desbalanceo de clases (94% positive). Un modelo ingenuo que siempre predice 'positive' obtiene 93.8% de accuracy — pero es inútil para el negocio. Resolver esto requirió entender que accuracy no es la métrica correcta, pasar a F1 por clase, e implementar class-weighted loss en el fine-tuning. El resultado fue un modelo con F1-negative de 0.73 frente a 0.45 del pretrained."

**P: ¿Por qué 4 modelos de clasificación?**  
R: "Empecé con RoBERTa como baseline razonable (entrenado en texto corto, 3 clases nativas). Al añadir nlptown descubrí que el *domain alignment* importa más que el tamaño del modelo: un modelo entrenado en reseñas de Amazon supera al entrenado en tweets sin ningún fine-tuning adicional (+2.7pp accuracy). El fine-tuning de ambos mostró que class weights ayudan al recall de minorías, pero la precisión queda limitada por el tamaño del dataset negativo (solo 108 muestras)."

**P: ¿Qué cambiarías si tuvieras más tiempo?**  
R: "Tres cosas: 1) Anotación manual de etiquetas — las etiquetas actuales vienen de ratings (1-5★), pero una reseña de 5★ puede tener críticas. 2) Añadir más reseñas negativas (o usar SMOTE específicamente en esa clase). 3) Evaluación cuantitativa de los artículos con ROUGE respecto a un conjunto de referencia escrito por humanos."

**P: ¿Cómo aseguras la reproducibilidad?**  
R: "`random_state=42` en todos los pasos estocásticos (PCA, KMeans, train/test split). El modelo RoBERTa es determinista en inferencia con `model.eval()`. El único componente no reproducible es GPT-4o-mini con temperature > 0 — cada regeneración producirá un artículo ligeramente distinto, lo cual es un comportamiento esperado y deseable."

**P: ¿Cómo está organizado el código para mantenimiento?**  
R: "Cada módulo (preprocess, classify, cluster, summarize) tiene su propia función `main()` ejecutable independientemente con `python src/modulo.py`. Las constantes están centralizadas al inicio de cada archivo. La app importa solo las funciones que necesita de cada módulo — nunca todo el módulo. Pylint 10/10 en todos los archivos."

---

## 7. Mapa de Conceptos — Bootcamp vs Proyecto

| Concepto del Bootcamp | Dónde aparece en el proyecto |
|---|---|
| Data preprocessing / EDA | `src/preprocess.py` — limpieza y análisis en `notebooks/01_eda.ipynb` |
| Supervised learning | `src/classify.py` — clasificación de sentimiento con Transformers |
| Class imbalance | `map_sentiment` → labels desbalanceados → class weights en fine-tuning |
| Transfer learning | Cargar RoBERTa/nlptown pretrained y fine-tunear con nuestros datos |
| Fine-tuning | `notebooks/colab/02_classification_finetuned_GPU.ipynb` |
| Unsupervised learning | `src/cluster.py` — K-Means sin etiquetas |
| Dimensionality reduction | `reduce_dimensions()` — PCA antes de K-Means |
| Evaluation metrics | Accuracy, Precision, Recall, F1, Silhouette, confusion matrix |
| Generative AI / LLMs | `src/summarize.py` — GPT-4o-mini con prompt engineering |
| Prompt engineering | `build_prompt()` — role + data + format structure |
| MLOps / deployment | `app/main.py` — Streamlit, caching, manejo de errores |
| API security | `.env` + `python-dotenv` — nunca hardcoded keys |

---

## 8. Chuleta Rápida — Números que debes memorizar

| Dato | Valor |
|---|---|
| Reseñas en dataset limpio | 4,805 |
| Distribución positivo/neutral/negativo | 93.8% / 4.0% / 2.2% |
| Modelos evaluados (clasificación) | 4 (2 pretrained + 2 fine-tuned) |
| Mejor accuracy (zero-shot) | nlptown pretrained — 91.53% |
| Mejor F1 negative | RoBERTa fine-tuned — 0.73 |
| Modelo seleccionado para producción | RoBERTa fine-tuned |
| Número de clusters | K=5, Silhouette=0.40 |
| Varianza explicada por PCA (10 comp.) | 89.15% |
| Artículos generados por GPT | 4 (1 categoría excluida — datos insuficientes) |
| Temperatura GPT | 0.7 |
| Tokens máximos GPT | 1,000 |
| Pylint score | 10.00 / 10 |
| Longitud máxima de tokens (Transformer) | 128 (cubre p95 de reseñas) |

---

## 9. La carpeta `notebooks/` — Estructura y Función

### La regla de oro

> **Notebooks = pensar y mostrar. `src/` = hacer y reutilizar.**

Los notebooks son documentos **narrativos e interactivos** — no están pensados para ejecutarse en producción, sino para ser leídos y presentados. El `src/` es la lógica limpia e importable que usa la app.

---

### Nivel 1 — Notebooks locales (análisis y resultados)

#### `01_eda.ipynb` — El punto de partida
Responde a una sola pregunta: **¿con qué datos trabajamos?**

Hace dos cosas en orden:
1. Carga el CSV **crudo** y documenta los 7 problemas que tiene (nulos, duplicados, categorías sucias, tipos incorrectos...) — justifica cada paso de `preprocess.py`
2. Carga el CSV **limpio** y analiza distribuciones: cómo se reparten los ratings, qué productos tienen más reseñas, cuánto miden los textos

Sin este notebook no sabrías por qué se tomaron las decisiones de limpieza ni qué te vas a encontrar en los modelos.

---

#### `02_classification.ipynb` — El juez de los modelos
**No entrena nada.** Evalúa y compara los resultados que vienen de los notebooks de Colab.

Carga los 4 CSV de predicciones (`reviews_with_predictions_*.csv`) y genera:
- Las confusion matrices de los 4 modelos
- Las tablas de Accuracy / F1 / Precision / Recall por modelo y clase
- La conclusión de qué modelo usar en producción y por qué

Sin este notebook tendrías predicciones en un CSV pero no sabrías si son buenas ni cuál elegir.

---

#### `03_clustering.ipynb` — El explorador de categorías
Agrupa los 23 productos en meta-categorías sin ninguna etiqueta previa (aprendizaje no supervisado).

Documenta el proceso de decisión:
- Por qué TF-IDF sobre el texto de las reseñas
- La curva elbow + silhouette para elegir K=5
- La visualización PCA 2D para verificar que los clusters tienen sentido visualmente
- Los nombres asignados a cada cluster ("Fire Tablets", "Alexa Devices"...)

Sin este notebook, el `cluster.py` existiría pero nadie entendería por qué K=5 y no K=4 o K=6.

---

#### `04_summarization.ipynb` — El ingeniero de prompts
Diseña y prueba el prompt que usa `summarize.py` para generar artículos con GPT-4o-mini.

Documenta:
- Por qué GPT-4o-mini y no BART o T5
- La estructura del prompt (5 secciones obligatorias)
- Ejemplos de salida para verificar la calidad antes de generar todos los artículos

Sin este notebook, `summarize.py` sería una caja negra — no sabrías qué instrucciones recibe el modelo ni por qué los artículos tienen esa estructura.

---

### Nivel 2 — Notebooks de Colab (ejecución en GPU)

Estos tres viven en `notebooks/colab/` porque **no se pueden ejecutar en un ordenador local** — necesitan una GPU de al menos 15 GB de VRAM para completarse en tiempo razonable.

| Notebook | Qué hace | Por qué en Colab |
|---|---|---|
| `02_classification_GPU.ipynb` | Inferencia zero-shot con RoBERTa pretrained sobre 4.805 reseñas | ~30 min en CPU, ~2 min en T4 GPU |
| `02_classification_finetuned_GPU.ipynb` | Fine-tuning de RoBERTa con class weights, 3 épocas | Imposible en CPU — backpropagation requiere GPU |
| `03_classification_nlptown_finetuned_GPU.ipynb` | Reemplaza la cabeza de clasificación de nlptown (5→3 clases) y entrena | Mismo motivo — fine-tuning con Trainer de HuggingFace |

El flujo es siempre el mismo: **subes el CSV limpio a Colab → ejecutas → descargas el CSV con predicciones → lo guardas en `data/processed/`**. El notebook local de evaluación (`02_classification.ipynb`) coge ese CSV y hace el análisis.

---

### El mapa completo

```
notebooks/
│
├── 01_eda.ipynb               ← ¿Qué tenemos? ¿Está limpio?
├── 02_classification.ipynb    ← ¿Qué modelo gana?  (lee resultados de Colab)
├── 03_clustering.ipynb        ← ¿Cuántos clusters? ¿Tienen sentido?
├── 04_summarization.ipynb     ← ¿El prompt funciona? ¿Los artículos son buenos?
│
└── colab/
    ├── 02_classification_GPU.ipynb               ← Inferencia RoBERTa (T4 GPU)
    ├── 02_classification_finetuned_GPU.ipynb      ← Fine-tuning RoBERTa (T4 GPU)
    └── 03_classification_nlptown_finetuned_GPU.ipynb  ← Fine-tuning nlptown (T4 GPU)
```

### Relación notebooks ↔ src/

```
Exploración (notebooks)              Producción (src/)
        │                                    │
01_eda.ipynb ──── descubrimientos ──► preprocess.py
02_classification.ipynb ────────────► classify.py  ──► app/main.py
03_clustering.ipynb ────────────────► cluster.py   ──► app/main.py
04_summarization.ipynb ─────────────► summarize.py ──► app/main.py
```

Los notebooks son el **"¿por qué?"**. El `src/` es el **"¿cómo?"** ejecutable. La app es el **"¿para quién?"**.

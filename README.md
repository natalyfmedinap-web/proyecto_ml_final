# Proyecto final · Aprendizaje de Máquina II

Réplica con datos simulados de:

> Carrillo Trujillo, M. V. y Aguilar Encarnación, P. S. (2025). *Análisis predictivo y gestión optimizada de la cartera de clientes morosos en CNEL EP, Unidad de Negocio Bolívar*. Revista Tecnológica ESPOL, 37(1), 285–308. https://doi.org/10.37815/rte.v37n1.1312

**Autora:** Nataly Medina · Maestría en Analítica Aplicada, Universidad de La Sabana

## Relación con el trabajo de grado

El trabajo de grado aborda la conciliación automática entre cuentas por pagar (AP) y cuentas por cobrar (AR). El artículo trata el lado AR: clasificar la cartera de clientes según su riesgo de morosidad con modelos ensemble.

## Estructura

```
proyecto_ml_final/
├── README.md
├── requirements.txt
├── data/            # se genera al correr los notebooks
├── notebooks/
│   ├── 01_simulacion_datos.ipynb       # simulación y verificación contra el paper
│   ├── 02_analisis_articulo.ipynb      # réplica del análisis y comparación
│   └── 03_extension_experimental.ipynb # eliminación de variables informativas
├── src/
│   ├── simulacion.py    # generación de la cartera y supuestos
│   ├── preparacion.py   # limpieza, winsorización, importancia RF, PCA + K-Means
│   ├── modelos.py       # modelos del paper, métricas, CV, estrés, extensión
│   ├── graficos.py      # matrices de confusión y comparaciones
│   ├── paper.py         # resultados reportados (Tablas 1, 2 y 3)
│   └── utilidades.py    # caché de resultados
├── results/         # tablas (.csv)
├── figures/         # gráficas (.png)
└── references/      # artículo en PDF
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Correr los notebooks **en orden** (01 → 02 → 03): cada uno usa los archivos que genera el anterior. Todo usa semilla 42, así que los resultados son reproducibles.

La validación cruzada, las pruebas de estrés y la extensión tardan (≈ 40 min en total con 2 núcleos, sobre todo por GradientBoosting). La primera vez se guardan en `results/`; al volver a correr, los notebooks los leen de ahí. Para recalcularlos, poner `RECALCULAR = True` en la primera celda de los notebooks 02 y 03.

## Procedimiento replicado

| Paso | Paper | Réplica |
|---|---|---|
| Datos | 72.483 registros reales, 43 variables | 72.483 clientes simulados, 17 variables |
| Limpieza | PASIVO, negativos, duplicados, moda en nulos | Igual |
| Atípicos | IQR + winsorización | IQR + winsorización P1–P99 (*supuesto*) |
| Selección | Random Forest → top 10 | Igual (objetivo: `NIVEL_MOROSIDAD`, *supuesto*) |
| Etiqueta | RobustScaler → PCA 90% → K-Means k=3 | log1p + RobustScaler 10–90 (*supuesto*) → PCA 90% → K-Means k=3 |
| Partición | 80/20, SMOTE solo en train | Igual (estratificada) |
| Modelos | RF, GB, AdaBoost, Voting (soft) | Igual, mismos hiperparámetros |
| Validación | CV estratificada k=5 | Igual (SMOTE dentro de cada fold) |
| Métricas | 9 métricas + matrices de confusión | Igual |
| Estrés | Ruido σ=0,3 · clase 2 al 50% · −40% datos | Igual |

Los supuestos de la simulación están en `src/simulacion.py` (`SUPUESTOS`) y se imprimen en el notebook 01.

## Extensión experimental

Se eliminan las k = 1, 3 y 5 variables más informativas de la entrada del clasificador, manteniendo las etiquetas originales. Evalúa cuánto del desempeño casi perfecto depende de que el clasificador use las mismas variables con las que K-Means creó las etiquetas.

## Referencia del artículo

El PDF está en `references/` y también en https://portal.amelica.org/ameli/journal/844/8445538016/8445538016.pdf (acceso abierto, CC BY-NC 4.0).

"""Preparación de datos y fase no supervisada (réplica de Carrillo y Aguilar, 2025)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

CATEGORICAS = ["CANTON", "TARIFA"]
CONTINUAS = ["VALOR_FACTURACION", "VALOR_RECAUDACION", "MESES_DEUDA", "V_VENCIDO_ENERGIA",
             "V_VENCIDO_ALUMBRADO", "V_VENCIDO_BOMBEROS", "V_VENCIDO_INTERESES", "VALOR_DEUDA"]
PREDICTORAS = CONTINUAS + ["RANGO_VALOR", "RANGO_ANTIGUEDAD", "CANTON", "TARIFA",
                           "TARIFA_DIGNIDAD", "DESC_TERCERA_EDAD"]


def limpiar(df):
    """Excluye PASIVO, deudas negativas y duplicados; imputa nulos categóricos con la moda."""
    reporte = {"inicial": len(df)}
    df = df[df.ESTADO != "PASIVO"]
    reporte["sin_pasivo"] = len(df)
    df = df[df.VALOR_DEUDA >= 0]
    reporte["sin_negativos"] = len(df)
    df = df.drop_duplicates()
    reporte["sin_duplicados"] = len(df)
    df = df.drop(columns=["ESTADO"]).copy()
    for col in CATEGORICAS:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df.reset_index(drop=True), reporte


def outliers_iqr(df, cols=CONTINUAS):
    """Cuenta atípicos por la regla de 1,5 x IQR."""
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    return ((df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)).sum()


def winsorizar(df, cols=CONTINUAS, inf=0.01, sup=0.99):
    """Recorta cada variable a sus percentiles 1 y 99 (supuesto: el paper no da los límites)."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].clip(df[c].quantile(inf), df[c].quantile(sup))
    return df


def codificar(df, cols=CATEGORICAS):
    """Label Encoding, como en el paper."""
    df = df.copy()
    for c in cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df


def importancia_rf(df, objetivo="NIVEL_MOROSIDAD", cols=PREDICTORAS, seed=42):
    """Importancia de variables con Random Forest (Figura 8 del paper)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(df[cols], df[objetivo])
    return pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)


def metodo_codo(X, ks=range(1, 9), seed=42):
    return pd.Series({k: KMeans(k, n_init=10, random_state=seed).fit(X).inertia_ for k in ks})


def matriz_clustering(df, cols, log=True, rango=(10, 90)):
    """log1p en montos (supuesto) + RobustScaler. Con log=False y rango=(25, 75) se obtiene la versión 'cruda'."""
    X = df[cols].astype(float).copy()
    if log:
        for c in cols:
            if c in CONTINUAS:
                X[c] = np.log1p(X[c])
    return RobustScaler(quantile_range=rango).fit_transform(X)


def etiquetar_riesgo(df, cols, k=3, varianza=0.90, seed=42, log=True, rango=(10, 90)):
    """RobustScaler -> PCA (90% var.) -> K-Means. Ordena clústeres por deuda media: 0 bajo, 1 medio, 2 alto."""
    X = matriz_clustering(df, cols, log, rango)
    pca = PCA(n_components=varianza, random_state=seed)
    Z = pca.fit_transform(X)
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    orden = df.groupby(km.labels_)["VALOR_DEUDA"].mean().sort_values().index
    etiquetas = pd.Series(km.labels_).map({c: i for i, c in enumerate(orden)}).values
    info = {
        "componentes": pca.n_components_,
        "varianza_explicada": pca.explained_variance_ratio_.sum(),
        "silhouette": silhouette_score(Z, km.labels_, sample_size=10_000, random_state=seed),
    }
    return etiquetas, Z, info

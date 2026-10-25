"""Fase supervisada: modelos con los hiperparámetros del paper y métricas de evaluación."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize


def construir_modelos(seed=42):
    """Hiperparámetros reportados en el paper. En GB, max_depth y min_samples_split son supuestos
    ("se limitaron los árboles base en profundidad y tamaño de partición", sin valores)."""
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=3,
                                max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8,
                                    max_depth=3, min_samples_split=10, random_state=seed)
    ada = AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=seed)
    return {"RandomForest": rf, "GradientBoosting": gb, "AdaBoost": ada}


class VotoSuave:
    """Votación suave con pesos iguales sobre modelos ya entrenados.
    Equivale a VotingClassifier(voting='soft') pero evita reentrenar los tres modelos base."""
    def __init__(self, modelos):
        self.modelos = list(modelos)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def ajustar(X, y, seed=42):
    """Aplica SMOTE, entrena RF, GB y AdaBoost y arma el VotingClassifier."""
    X_sm, y_sm = SMOTE(random_state=seed).fit_resample(X, y)
    modelos = construir_modelos(seed)
    for m in modelos.values():
        m.fit(X_sm, y_sm)
    modelos["VotingClassifier"] = VotoSuave(modelos.values())
    return modelos


def metricas(y, pred, proba):
    """Las 9 métricas de la Tabla 2 del paper."""
    y_bin = label_binarize(y, classes=[0, 1, 2])
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "F1 Macro": f1_score(y, pred, average="macro"),
        "Precision Macro": precision_score(y, pred, average="macro", zero_division=0),
        "Recall Macro": recall_score(y, pred, average="macro"),
        "Cohen's Kappa": cohen_kappa_score(y, pred),
        "Matthews Corrcoef": matthews_corrcoef(y, pred),
        "AUC ROC (OVR)": roc_auc_score(y, proba, multi_class="ovr"),
        "PR AUC": average_precision_score(y_bin, proba, average="macro"),
    }


def entrenar_evaluar(X, y, seed=42):
    """Partición 80/20 estratificada, SMOTE solo en entrenamiento, entrena y evalúa cada modelo."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    modelos = ajustar(X_tr, y_tr, seed)
    filas, matrices = {}, {}
    for nombre, modelo in modelos.items():
        pred, proba = modelo.predict(X_te), modelo.predict_proba(X_te)
        filas[nombre] = metricas(y_te, pred, proba)
        matrices[nombre] = confusion_matrix(y_te, pred)
    return pd.DataFrame(filas).T, matrices, modelos


def validacion_cruzada(X, y, k=5, seed=42):
    """Accuracy promedio con validación cruzada estratificada (SMOTE se aplica dentro de cada fold)."""
    X, y = pd.DataFrame(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scores = {}
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X_tr, y_tr):
        modelos = ajustar(X_tr.iloc[tr], y_tr.iloc[tr], seed)
        for nombre, m in modelos.items():
            scores.setdefault(nombre, []).append(accuracy_score(y_tr.iloc[va], m.predict(X_tr.iloc[va])))
    return pd.DataFrame({"CV Accuracy": {n: np.mean(s) for n, s in scores.items()},
                         "CV Std": {n: np.std(s) for n, s in scores.items()}})


def escenarios_estres(X, y, seed=42):
    """Los 4 escenarios del paper: base, ruido sigma=0,3, clase 2 al 50%, eliminación del 40%.
    Supuesto: el ruido es gaussiano con sigma = 0,3 x desviación estándar de cada variable."""
    rng = np.random.default_rng(seed)
    X, y = pd.DataFrame(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    idx2 = y[y == 2].index
    quitar2 = rng.choice(idx2, len(idx2) // 2, replace=False)
    conservar = rng.choice(len(y), int(0.6 * len(y)), replace=False)
    escenarios = {
        "Base (original)": (X, y),
        "Ruido moderado (σ=0.3)": (X + rng.normal(0, 0.3, X.shape) * X.std().values, y),
        "Reducción clase 2 al 50%": (X.drop(quitar2), y.drop(quitar2)),
        "Eliminación aleatoria del 40%": (X.iloc[conservar], y.iloc[conservar]),
    }
    tablas = []
    for nombre, (Xe, ye) in escenarios.items():
        res, _, _ = entrenar_evaluar(Xe, ye, seed)
        tablas.append(res.assign(Escenario=nombre))
    return pd.concat(tablas).rename_axis("Modelo").reset_index()


def extension_sin_variables(X, y, ranking, ks=(1, 3, 5), seed=42):
    """Extensión experimental: elimina las k variables más informativas y repite el mismo procedimiento.
    Las etiquetas de riesgo NO cambian (se crearon con todas las variables)."""
    tablas = []
    for k in ks:
        quitar = list(ranking[:k])
        res, _, _ = entrenar_evaluar(X.drop(columns=quitar), y, seed)
        tablas.append(res.assign(Variables_eliminadas=k, Eliminadas=", ".join(quitar)))
    return pd.concat(tablas).rename_axis("Modelo").reset_index()

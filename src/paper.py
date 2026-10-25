"""Resultados reportados por Carrillo y Aguilar (2025), Tablas 1, 2 y 3."""
import pandas as pd

COLS = ["Accuracy", "Balanced Accuracy", "F1 Macro", "Precision Macro", "Recall Macro",
        "Cohen's Kappa", "Matthews Corrcoef", "AUC ROC (OVR)", "PR AUC"]

TABLA_2 = pd.DataFrame({
    "RandomForest":     [0.9965, 0.9980, 0.9916, 0.9853, 0.9980, 0.9873, 0.9874, 1.0, 0.9998],
    "GradientBoosting": [0.9982, 0.9987, 0.9957, 0.9927, 0.9987, 0.9935, 0.9935, 1.0, 1.0],
    "AdaBoost":         [0.9845, 0.9805, 0.9590, 0.9404, 0.9805, 0.9450, 0.9459, 0.9997, 0.9981],
    "VotingClassifier": [0.9983, 0.9990, 0.9959, 0.9927, 0.9990, 0.9938, 0.9938, 1.0, 0.9999],
}, index=COLS).T

TABLA_1_CV = pd.Series({"RandomForest": 0.999, "GradientBoosting": 0.999,
                        "AdaBoost": 0.983, "VotingClassifier": 0.999}, name="CV Accuracy")

# Tabla 3: F1 Macro por escenario de estrés
TABLA_3_F1 = pd.DataFrame({
    "Base (original)":               [0.9916, 0.9957, 0.9590, 0.9959],
    "Ruido moderado (σ=0.3)":        [0.9823, 0.9865, 0.9517, 0.9885],
    "Reducción clase 2 al 50%":      [0.9886, 0.9943, 0.9390, 0.9943],
    "Eliminación aleatoria del 40%": [0.9917, 0.9962, 0.9610, 0.9962],
}, index=["RandomForest", "GradientBoosting", "AdaBoost", "VotingClassifier"])

CLUSTERING = {"varianza_explicada": 0.9021, "silhouette": 0.681}

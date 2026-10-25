"""Simulación de la cartera de clientes de CNEL EP – Unidad de Negocio Bolívar."""
import numpy as np
import pandas as pd

SUPUESTOS = [
    "El 13,6% de cartera vencida se interpreta como % de clientes con >= 2 meses de deuda.",
    "Proporciones de cantón y tarifa supuestas (no reportadas).",
    "Facturación lognormal con medianas 15 / 60 / 400 USD (residencial / comercial / industrial).",
    "La mora depende de zona rural, subsidio y factura relativa, más ruido individual.",
    "Meses de mora: 2 + binomial negativa, recortada en 193 (máximo del paper).",
    "Valores vencidos calculados a partir de factura x meses; deuda recortada en 27.854 USD.",
    "Se simulan 17 de las 43 variables (las que el paper nombra o se requieren).",
    "RANGO_PASTEL no está definido en el paper; se reemplaza por RANGO_ANTIGUEDAD.",
    "Suciedad: 3% PASIVO, 0,5% deudas negativas, 1% nulos en CANTON/TARIFA, 0,3% duplicados.",
]

CANTONES = ["GUARANDA", "SAN MIGUEL", "CHIMBO", "CHILLANES", "CALUMA", "ECHEANDIA", "LAS NAVES"]
P_CANTON = [0.42, 0.14, 0.12, 0.11, 0.09, 0.08, 0.04]
TARIFAS = ["RESIDENCIAL", "COMERCIAL", "INDUSTRIAL"]
P_TARIFA = [0.85, 0.12, 0.03]
MEDIANA_FACT = {"RESIDENCIAL": 15, "COMERCIAL": 60, "INDUSTRIAL": 400}
RURALES = ["CHILLANES", "ECHEANDIA", "LAS NAVES"]


def simular_cartera(n=72_483, pct_mora=0.136, seed=42):
    """Genera la cartera limpia."""
    rng = np.random.default_rng(seed)

    # 1. Categóricas
    canton = rng.choice(CANTONES, n, p=P_CANTON)
    tarifa = rng.choice(TARIFAS, n, p=P_TARIFA)
    es_res = tarifa == "RESIDENCIAL"
    dignidad = (es_res & (rng.random(n) < 0.30)).astype(int)
    tercera_edad = (es_res & (rng.random(n) < 0.08)).astype(int)

    # 2. Facturación (sesgada a la derecha)
    mediana = np.array([MEDIANA_FACT[t] for t in tarifa]) * np.where(dignidad == 1, 0.6, 1.0)
    facturacion = rng.lognormal(np.log(mediana), 0.6)

    # 3. Propensión latente a la mora
    propension = (rng.normal(0, 1.2, n) + 0.4 * np.isin(canton, RURALES)
                  - 0.3 * dignidad + 0.2 * np.log(facturacion / mediana))
    moroso = propension > np.quantile(propension, 1 - pct_mora)

    # 4. Meses de deuda y valores vencidos
    meses = np.where(moroso, 2 + rng.negative_binomial(0.5, 0.03, n), rng.choice([0, 1], n, p=[0.7, 0.3]))
    meses = np.clip(meses, 0, 193)
    v_energia = facturacion * meses * rng.uniform(0.85, 1.0, n)
    v_alumbrado = v_energia * rng.uniform(0.08, 0.12, n)
    v_bomberos = 0.5 * meses
    v_intereses = v_energia * 0.008 * meses
    deuda = np.clip(facturacion + v_energia + v_alumbrado + v_bomberos + v_intereses, 0, 27_854)
    pago_ok = rng.uniform(0.9, 1.1, n) * facturacion
    pago_parcial = rng.uniform(0, 0.5, n) * facturacion * (rng.random(n) < 0.4)
    recaudacion = np.where(moroso, pago_parcial, pago_ok)

    df = pd.DataFrame({
        "ID_CLIENTE": np.arange(1, n + 1), "CANTON": canton, "TARIFA": tarifa,
        "TARIFA_DIGNIDAD": dignidad, "DESC_TERCERA_EDAD": tercera_edad, "ESTADO": "ACTIVO",
        "VALOR_FACTURACION": facturacion.round(2), "VALOR_RECAUDACION": recaudacion.round(2),
        "MESES_DEUDA": meses, "V_VENCIDO_ENERGIA": v_energia.round(2),
        "V_VENCIDO_ALUMBRADO": v_alumbrado.round(2), "V_VENCIDO_BOMBEROS": v_bomberos.round(2),
        "V_VENCIDO_INTERESES": v_intereses.round(2), "VALOR_DEUDA": deuda.round(2),
    })
    df["RANGO_VALOR"] = pd.cut(df.VALOR_DEUDA, [-1, 20, 50, 100, 250, 500, 1000, 5000, np.inf], labels=False) + 1
    df["RANGO_ANTIGUEDAD"] = pd.cut(df.MESES_DEUDA, [-1, 1, 3, 6, 12, 24, np.inf], labels=False) + 1
    df["NIVEL_MOROSIDAD"] = pd.cut(df.MESES_DEUDA, [-1, 1, 3, np.inf],
                                   labels=["AL_DIA", "MORA_CORTA", "MORA_LARGA"]).astype(str)
    return df


def ensuciar(df, seed=42):
    """Agrega los problemas de calidad que el paper limpia."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "ESTADO"] = "PASIVO"
    df.loc[rng.choice(n, int(0.005 * n), replace=False), "VALOR_DEUDA"] *= -1
    for col in ["CANTON", "TARIFA"]:
        df.loc[rng.choice(n, int(0.01 * n), replace=False), col] = np.nan
    dups = df.sample(int(0.003 * n), random_state=seed)
    return pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

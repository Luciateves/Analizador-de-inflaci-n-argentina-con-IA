"""Lógica del Analizador de Inflación Argentina IA (separada de la interfaz)."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import warnings
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

API_URL = "https://apis.datos.gob.ar/series/api/series/"
API_ID = "148.3_INIVELNAL_DICI_M_26"
MIRROR_URL = "https://ncachanosky.github.io/argentina-economic-monitor/data/cpi.json"
MODEL_ORDERS = {
    "ARIMA(1,0,0)": (1, 0, 0),
    "ARIMA(2,0,0)": (2, 0, 0),
    "ARIMA(1,0,1)": (1, 0, 1),
}


@dataclass
class DataSet:
    df: pd.DataFrame
    fuente: str
    url: str
    simulado: bool = False


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe columnas fecha/ipc y exige una serie mensual ordenada y completa."""
    if not {"fecha", "ipc"}.issubset(df.columns):
        raise ValueError("El CSV debe contener columnas 'fecha' e 'ipc'.")
    df = df[["fecha", "ipc"]].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["ipc"] = pd.to_numeric(df["ipc"], errors="coerce")
    df = df.dropna().drop_duplicates("fecha", keep="last").sort_values("fecha")
    df = df[df["ipc"] > 0]
    if len(df) < 38:
        raise ValueError("Se necesitan al menos 38 meses de IPC para análisis y pronóstico.")
    # Fechas normalizadas al comienzo del mes: la API publica YYYY-MM-01.
    df["fecha"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df = df.drop_duplicates("fecha", keep="last").set_index("fecha").asfreq("MS")
    if df["ipc"].isna().any():
        faltan = df.index[df["ipc"].isna()]
        raise ValueError("La serie tiene meses sin datos: " + ", ".join(faltan.strftime("%m/%Y")[:5]))
    return df


def descargar_datos(session=None) -> DataSet:
    """Consulta 2 fuentes con timeout; escoge la que tenga el último período."""
    ses = session or requests
    validos: list[DataSet] = []
    errores = []
    try:
        r = ses.get(API_URL, params={"ids": API_ID, "limit": 1000, "metadata": "none"}, timeout=(5, 15))
        r.raise_for_status()
        datos = r.json()["data"]
        df = normalizar(pd.DataFrame(datos, columns=["fecha", "ipc"]))
        validos.append(DataSet(df, "API de Series de Tiempo / IPC del INDEC", API_URL + "?ids=" + API_ID))
    except Exception as ex:
        errores.append(f"API oficial: {str(ex)[:130]}")
    try:
        r = ses.get(MIRROR_URL, timeout=(5, 15))
        r.raise_for_status()
        j = r.json()
        fechas = j["dates"]
        valores = j["variants"]["headline"]["values"]
        if len(fechas) != len(valores):
            raise ValueError("Los datos del espejo están incompletos")
        df = normalizar(pd.DataFrame({"fecha": fechas, "ipc": valores}))
        validos.append(DataSet(df, "Espejo de la serie INDEC (Argentina Economic Monitor)", MIRROR_URL))
    except Exception as ex:
        errores.append(f"Espejo: {str(ex)[:130]}")
    if not validos:
        raise ConnectionError("No fue posible descargar el IPC. " + " | ".join(errores))
    return max(validos, key=lambda d: (d.df.index.max(), len(d.df), int("API de" in d.fuente)))


def desde_csv(contenido: bytes) -> DataSet:
    """Archivo propio de usuario: fecha,ipc con fecha ISO. No se presume oficial."""
    ultimo_error = None
    for codificacion in ["utf-8-sig", "latin-1"]:
        for sep in [",", ";"]:
            try:
                datos = pd.read_csv(BytesIO(contenido), sep=sep, encoding=codificacion)
                datos.columns = [x.strip().lower() for x in datos.columns]
                return DataSet(normalizar(datos), "Archivo CSV proporcionado por usuario (no verificado)", "")
            except Exception as ex:
                ultimo_error = ex
    raise ValueError(f"No se pudo leer el CSV. Debe tener fecha,ipc. {ultimo_error}")


def crear_demo() -> DataSet:
    """Solamente para ensayar la interfaz sin internet. Son datos FICTICIOS."""
    rng = np.random.default_rng(12)
    fechas = pd.date_range("2017-01-01", periods=118, freq="MS")
    t = np.arange(len(fechas))
    mensual = 2.1 + 0.70 * np.sin(t / 5.0) + 0.35 * rng.normal(size=len(t))
    mensual += 6.0 * np.exp(-((t - 77) / 9) ** 2)
    mensual = np.maximum(0.15, mensual)
    indices = 100 * np.cumprod(1 + mensual / 100)
    df = pd.concat([pd.DataFrame({"fecha": [pd.Timestamp("2016-12-01")], "ipc": [100.0]}),
                    pd.DataFrame({"fecha": fechas, "ipc": indices})], ignore_index=True)
    return DataSet(normalizar(df), "DATOS SIMULADOS / solo demostración", "", True)


def indicadores(ipc: pd.DataFrame) -> pd.DataFrame:
    df = ipc.copy()
    df["mensual"] = df.ipc.pct_change(fill_method=None) * 100
    df["interanual"] = df.ipc.pct_change(periods=12, fill_method=None) * 100
    base_dic = df.ipc.where(df.index.month == 12).ffill().shift(1)
    df["acumulada"] = (df.ipc / base_dic - 1) * 100
    df["media_3m"] = df.mensual.rolling(3).mean()
    return df


def anomalias(serie: pd.Series, contamination: float = 0.07) -> pd.DataFrame:
    datos = serie.dropna().to_frame("mensual")
    if len(datos) < 24:
        return pd.DataFrame(columns=["mensual", "atipico"])
    modelo = IsolationForest(n_estimators=120, contamination=contamination, random_state=42)
    datos["atipico"] = modelo.fit_predict(datos[["mensual"]]) == -1
    return datos


def calcular_pronostico(serie: pd.Series, horizonte: int = 6) -> dict:
    """Comparación fuera de muestra (12 meses) y reestimación con toda la serie."""
    s = serie.dropna().asfreq("MS").astype(float)
    if len(s) < 36:
        raise ValueError("Se necesitan al menos 36 observaciones de inflación mensual.")
    h = 12
    train, test = s.iloc[:-h], s.iloc[-h:]
    filas = []
    modelos = {}
    ingenuo = np.repeat(train.iloc[-1], h)
    filas.append({"modelo": "Persistencia (referencia)", "MAE": float(mean_absolute_error(test, ingenuo)),
                  "AIC": np.nan, "BIC": np.nan})
    for nombre, orden in MODEL_ORDERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(train, order=orden, enforce_stationarity=False,
                            enforce_invertibility=False).fit(method_kwargs={"maxiter": 65})
            pred = np.asarray(fit.forecast(steps=h))
            if not np.all(np.isfinite(pred)):
                continue
            filas.append({"modelo": nombre, "MAE": float(mean_absolute_error(test, pred)),
                          "AIC": float(fit.aic), "BIC": float(fit.bic)})
            modelos[nombre] = fit
        except Exception:
            continue
    tabla = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)
    ganador = tabla.loc[0, "modelo"]
    fechas = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS")
    if ganador not in MODEL_ORDERS:
        futuro = np.repeat(s.iloc[-1], horizonte)
        intervalos = None
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_final = ARIMA(s, order=MODEL_ORDERS[ganador], enforce_stationarity=False,
                                  enforce_invertibility=False).fit(method_kwargs={"maxiter": 65})
            fc = fit_final.get_forecast(steps=horizonte)
            futuro = np.asarray(fc.predicted_mean)
            intervalos = np.asarray(fc.conf_int(alpha=0.20))
        except Exception:
            ganador = "Persistencia (referencia)"
            futuro = np.repeat(s.iloc[-1], horizonte)
            intervalos = None
    pron = pd.DataFrame({"fecha": fechas, "inflacion_proyectada": futuro})
    if intervalos is not None:
        pron["limite_inferior_80"] = intervalos[:, 0]
        pron["limite_superior_80"] = intervalos[:, 1]
    diag = None
    if modelos:
        nombre_diag = min(modelos, key=lambda n: next(f["MAE"] for f in filas if f["modelo"] == n))
        try:
            residuo = modelos[nombre_diag].resid.dropna()
            if len(residuo) > 14:
                diag = {"modelo": nombre_diag,
                        "p_ljung_box_12": float(acorr_ljungbox(residuo, lags=[12], return_df=True).lb_pvalue.iloc[0])}
        except Exception:
            pass
    return {"comparacion": tabla, "mejor": ganador, "pronostico": pron, "diagnostico": diag,
            "mae": float(tabla.iloc[0]["MAE"]), "train_end": train.index.max(),
            "test_start": test.index.min(), "test_end": test.index.max()}


def resumen(df: pd.DataFrame, resultado: dict, outliers: pd.DataFrame, fuente: str) -> str:
    last = df.iloc[-1]
    reciente = df.mensual.iloc[-3:].mean()
    previo = df.mensual.iloc[-6:-3].mean()
    diferencia = reciente - previo
    tendencia = ("aceleración" if diferencia > 0.15 else
                 "desaceleración" if diferencia < -0.15 else "relativa estabilidad")
    def fmt(x):
        return f"{x:.2f}".replace(".", ",") if pd.notna(x) else "no disponible"
    return (
        f"Informe del IPC - {df.index[-1]:%m/%Y}. Fuente: {fuente}. "
        f"Inflación mensual: {fmt(last.mensual)}%; interanual: {fmt(last.interanual)}%; "
        f"acumulada desde diciembre anterior: {fmt(last.acumulada)}%. "
        f"El promedio de los últimos tres meses fue {fmt(reciente)}%, "
        f"frente a {fmt(previo)}% en los tres meses anteriores; esto indica {tendencia} "
        f"según una regla descriptiva. Isolation Forest detectó {int(outliers.atipico.sum())} "
        f"meses atípicos en la muestra. Entre los modelos evaluados con los últimos 12 meses "
        f"separados como prueba, el de menor MAE fue {resultado['mejor']} "
        f"({fmt(resultado['mae'])} puntos porcentuales). "
        f"La proyección para el próximo mes es {fmt(resultado['pronostico'].iloc[0].inflacion_proyectada)}%. "
        "El pronóstico es condicional, no garantiza valores futuros ni explica causalidad."
    )


def calcular_variacion_real(monto: float, inflacion_porcentual: float, aumento_porcentual: float = 0.0) -> dict:
    final_nominal = monto * (1 + aumento_porcentual / 100)
    final_real = final_nominal / (1 + inflacion_porcentual / 100)
    return {"nominal": final_nominal, "real": final_real,
            "variacion_real": 100 * (final_real / monto - 1)}


def pdf_resumen(df: pd.DataFrame, datos: DataSet, resultado: dict, texto: str) -> bytes:
    """Informe breve generado desde la app, máximo 2 páginas."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=48, leftMargin=48, topMargin=45, bottomMargin=45)
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="TituloApp", parent=s["Title"], fontName="Helvetica-Bold", fontSize=18,
                         textColor=colors.HexColor("#172d52"), spaceAfter=16))
    s.add(ParagraphStyle(name="CuerpoApp", parent=s["Normal"], fontSize=10.3, leading=16, spaceAfter=12))
    story = [Paragraph("Analizador de Inflacion Argentina IA", s["TituloApp"]),
             Paragraph("Informe generado por la aplicacion - datos al " + df.index.max().strftime("%m/%Y"), s["CuerpoApp"]),
             Paragraph("1. Problema y objetivo", s["Heading2"]),
             Paragraph("Analizar el comportamiento del indice de precios al consumidor de Argentina "
                       "y elaborar proyecciones y alertas de meses atipicos mediante tecnicas estadisticas y de aprendizaje automatico.", s["CuerpoApp"]),
             Paragraph("2. Metodologia", s["Heading2"]),
             Paragraph("A partir de la serie mensual del indice IPC, se calculan variaciones mensual, "
                       "interanual y acumulada. Isolation Forest identifica valores inusuales de inflacion mensual. "
                       "Se comparan una referencia de persistencia y modelos ARIMA con los 12 ultimos meses "
                       "como conjunto de prueba; se selecciona el menor MAE y se generan pronosticos.", s["CuerpoApp"]),
             Paragraph("3. Resultados principales", s["Heading2"])]
    last = df.iloc[-1]
    tabla = Table([["Indicador", "Resultado"],
                   ["Inflacion mensual", f"{last.mensual:.2f}%"],
                   ["Inflacion interanual", f"{last.interanual:.2f}%"],
                   ["Inflacion acumulada", f"{last.acumulada:.2f}%"],
                   ["Modelo mejor evaluado", str(resultado["mejor"])],
                   ["Error MAE (12 meses)", f"{resultado['mae']:.2f} pp"],
                   ["Pronostico proximo mes", f"{resultado['pronostico'].iloc[0].inflacion_proyectada:.2f}%"]],
                  colWidths=[240, 230], hAlign="LEFT")
    tabla.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172d52")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5deea")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([tabla, Spacer(1, 15), Paragraph("4. Interpretacion automatica", s["Heading2"]),
                  Paragraph(escape(texto), s["CuerpoApp"]),
                  Paragraph("5. Limitaciones y fuente", s["Heading2"]),
                  Paragraph("Fuente: " + escape(datos.fuente) + ". La deteccion de anomalias "
                            "no establece causas; ARIMA extrapola patrones y no incorpora explicitamente "
                            "politica economica, expectativas ni shocks futuros. "
                            "Los resultados simulados, si se selecciono ese modo, NO corresponden a INDEC.", s["CuerpoApp"])])
    doc.build(story)
    return buffer.getvalue()

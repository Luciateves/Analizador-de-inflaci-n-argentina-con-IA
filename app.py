"""Aplicación web del parcial: Analizador de Inflación Argentina IA.
Para ejecutar: streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

from inflacion_core import (
    API_ID, DataSet, anomalias, calcular_pronostico, calcular_variacion_real,
    crear_demo, descargar_datos, desde_csv, indicadores, pdf_resumen, resumen,
)

st.set_page_config(page_title="Inflación Argentina IA | Analizador", page_icon="🇦🇷",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  .block-container { padding-top: 1.7rem; max-width: 1240px; }
  .stApp { background-color: #f7f9fc; }
  [data-testid="stMetric"] {background:white;border:1px solid #e5eaf3;border-radius:14px;padding:16px 18px;}
  [data-testid="stMetricLabel"] {color:#61718c;}
  [data-testid="stSidebar"] {background:#eff3f9;}
  h1, h2, h3 {color: #192c4b; letter-spacing:-.02em;}
  .section-label {color:#64748b;text-transform:uppercase;letter-spacing:.13em;font-weight:700;font-size:.8rem;}
  .hero {background:linear-gradient(115deg,#142d52 0%,#16436f 73%,#166d95 100%);color:white;border-radius:20px;padding:30px 32px;margin:8px 0 24px;}
  .hero h1 {color:white;font-weight:750;margin:3px 0 8px;font-size:2.15rem;}
  .hero p {color:#cddfed;margin:0;}
  .note {color:#63718a;font-size:.9rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_online() -> DataSet:
    return descargar_datos()

@st.cache_data(ttl=1800, show_spinner=False)
def modelos_cache(valores: tuple[float, ...], dates: tuple[str, ...], horizonte: int) -> dict:
    serie = pd.Series(valores, index=pd.to_datetime(list(dates)))
    return calcular_pronostico(serie, horizonte)

@st.cache_data(ttl=1800, show_spinner=False)
def anomalias_cache(valores: tuple[float, ...], dates: tuple[str, ...]) -> pd.DataFrame:
    serie = pd.Series(valores, index=pd.to_datetime(list(dates)))
    return anomalias(serie)


def fmt(valor: float, dec: int = 2) -> str:
    if pd.isna(valor):
        return "—"
    return f"{valor:,.{dec}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def base_chart(titulo: str, eje: str = "Variación (%)") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title={"text": titulo, "font": {"size": 19, "color": "#1c3152"}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", font={"family": "Arial", "color": "#334155"},
        hovermode="x unified", margin=dict(l=10, r=10, t=52, b=16), height=390,
        xaxis=dict(showgrid=False, title="Fecha"), yaxis=dict(title=eje, gridcolor="#e8eef5"),
        legend=dict(orientation="h", y=-0.24, x=0))
    return fig


def obtener_ia(texto: str, clave: str) -> str:
    """Petición opcional; no es necesaria para operar la aplicación."""
    prompt = ("Actuá como economista. Redactá en español claro un resumen económico de hasta "
              "180 palabras. Usá EXCLUSIVAMENTE los datos siguientes; no inventes fechas, "
              "cifras ni explicaciones causales. Separá valores observados de pronósticos, "
              "aclará limitaciones y evitá dar asesoramiento financiero.\n\n" + texto)
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}],
                     "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}},
                      headers={"x-goog-api-key": clave, "Content-Type": "application/json"},
                      timeout=(5, 20))
    r.raise_for_status()
    partes = r.json()["candidates"][0]["content"]["parts"]
    salida = "\n".join(p.get("text", "") for p in partes).strip()
    if not salida:
        raise ValueError("Gemini no devolvió contenido de texto.")
    return salida


with st.sidebar:
    st.markdown("### 🇦🇷 Inflación Argentina IA")
    st.caption("Segundo parcial · Programación para Economistas")
    st.divider()
    pagina = st.radio("Explorar", ["📊 Panorama", "📈 Series históricas", "🤖 IA y pronósticos",
                                      "💵 Poder adquisitivo", "📝 Informe y descargas"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Origen de los datos**")
    modo = st.radio("Fuente", ["Datos oficiales en línea", "Cargar un CSV", "Demo con datos simulados"],
                    label_visibility="collapsed")
    archivo = None
    if modo == "Cargar un CSV":
        archivo = st.file_uploader("Archivo CSV con columnas fecha,ipc", type=["csv"])
        st.caption("Por ejemplo: 2026-01-01,1234.5")
    if modo == "Datos oficiales en línea":
        if st.button("↻ Actualizar fuente", use_container_width=True):
            cargar_online.clear()
            st.rerun()
    st.divider()
    st.caption("IPC nacional: base diciembre 2016 = 100. Los resultados son descriptivos y no constituyen recomendaciones.")

try:
    if modo == "Datos oficiales en línea":
        with st.spinner("Consultando series económicas..."):
            conjunto = cargar_online()
    elif modo == "Cargar un CSV":
        if archivo is None:
            st.info("Subí un archivo CSV con las columnas `fecha` e `ipc` para comenzar.")
            st.stop()
        conjunto = desde_csv(archivo.getvalue())
    else:
        conjunto = crear_demo()
except Exception as error:
    st.error("No se pudo cargar la serie: " + str(error))
    st.info("Para probar la aplicación sin internet, seleccioná 'Demo con datos simulados' en el menú lateral. "
            "Para trabajar con datos reales, elegí 'Datos oficiales en línea' o cargá un CSV del IPC.")
    st.stop()

df = indicadores(conjunto.df)
serie = df.mensual.dropna()
ultimo = df.iloc[-1]
mes_final = df.index[-1].strftime("%m/%Y")

st.markdown('<div class="hero"><div class="section-label" style="color:#9bd5fa">OBSERVATORIO ECONÓMICO</div>'
            '<h1>Analizador de Inflación Argentina</h1>'
            '<p>Datos · Visualización · Machine Learning · Proyección económica</p></div>',
            unsafe_allow_html=True)
if conjunto.simulado:
    st.warning("🧪 MODO DEMOSTRACIÓN: los datos son SIMULADOS y NO corresponden al IPC real del INDEC.")
elif modo == "Cargar un CSV":
    st.warning("El CSV fue cargado por el usuario: su origen y exactitud no se verificaron automáticamente.")
else:
    edad = pd.Timestamp.today().to_period("M").ordinal - df.index[-1].to_period("M").ordinal
    if edad > 2:
        st.warning(f"El último dato descargado es {mes_final}; verificá si existen publicaciones más recientes.")

st.caption(f"Fuente: **{conjunto.fuente}** · Última observación: **{mes_final}** · {len(df)} registros mensuales")

if pagina in ["📊 Panorama", "📈 Series históricas"]:
    a, b, c = st.columns(3)
    a.metric("Inflación mensual", fmt(ultimo.mensual) + " %", help="Variación del IPC respecto del mes anterior")
    b.metric("Inflación interanual", fmt(ultimo.interanual) + " %", help="Variación del IPC respecto del mismo mes del año anterior")
    c.metric("Inflación acumulada", fmt(ultimo.acumulada) + " %", help="Variación desde diciembre del año anterior")

if pagina == "📊 Panorama":
    st.subheader("Evolución reciente")
    rango = st.slider("Período para visualizar (meses)", min_value=12,
                      max_value=min(96, len(df)), value=min(36, len(df)), step=12)
    graf = base_chart("Inflación mensual y promedio móvil (3 meses)")
    d = df.tail(rango)
    graf.add_trace(go.Scatter(x=d.index, y=d.mensual, mode="lines+markers", name="Inflación mensual",
                              line=dict(color="#178ad6", width=3), marker=dict(size=6)))
    graf.add_trace(go.Scatter(x=d.index, y=d.media_3m, mode="lines", name="Media móvil 3m",
                              line=dict(color="#eea44c", width=3, dash="dot")))
    st.plotly_chart(graf, use_container_width=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### ¿Qué significan estos indicadores?")
        st.markdown("**Mensual:** `(IPC_t / IPC_(t-1) - 1) × 100`.  \n"
                    "**Interanual:** `(IPC_t / IPC_(t-12) - 1) × 100`.  \n"
                    "**Acumulada:** variación desde diciembre del año anterior.")
    with right:
        ultimos = df[["mensual", "interanual", "acumulada"]].tail(6).rename(columns={
            "mensual": "Mensual (%)", "interanual": "Interanual (%)", "acumulada": "Acumulada (%)"})
        ultimos.index = ultimos.index.strftime("%m/%Y")
        st.dataframe(ultimos.style.format("{:.2f}"), use_container_width=True)

elif pagina == "📈 Series históricas":
    st.subheader("Explorador de la serie del IPC")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        elegida = st.selectbox("Indicador", ["Inflación mensual", "Inflación interanual",
                                               "Inflación acumulada del año", "Índice IPC"])
    with c2:
        ventana = st.slider("Últimos meses", min_value=12, max_value=len(df), value=min(60, len(df)))
    variables = {"Inflación mensual": "mensual", "Inflación interanual": "interanual",
                 "Inflación acumulada del año": "acumulada", "Índice IPC": "ipc"}
    col = variables[elegida]
    graf = base_chart(elegida + " · Argentina", "Índice (dic. 2016 = 100)" if col == "ipc" else "%")
    d = df.tail(ventana)
    graf.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=elegida,
                              fill="tozeroy" if col == "mensual" else None,
                              line=dict(color="#157da9", width=3)))
    st.plotly_chart(graf, use_container_width=True)
    st.markdown("#### Tabla histórica")
    tabla = df[["ipc", "mensual", "interanual", "acumulada"]].copy()
    tabla.index.name = "fecha"
    st.dataframe(tabla.tail(36).sort_index(ascending=False).style.format("{:.2f}"), use_container_width=True)
    st.download_button("⬇ Descargar la serie completa (CSV)", tabla.to_csv().encode("utf-8-sig"),
                       file_name="serie_ipc_analizada.csv", mime="text/csv")

elif pagina == "🤖 IA y pronósticos":
    st.subheader("Machine Learning y pronóstico económico")
    st.markdown("El algoritmo **Isolation Forest** reconoce meses estadísticamente atípicos. "
                "Después comparamos modelos ARIMA con una referencia sencilla mediante una prueba fuera de muestra.")
    with st.spinner("Analizando la serie..."):
        atipicos = anomalias_cache(tuple(serie.values.tolist()), tuple(serie.index.astype(str)))
    izquierda, derecha = st.columns([1, 1])
    with izquierda:
        st.metric("Meses atípicos detectados", int(atipicos.atipico.sum()))
    with derecha:
        meses = st.select_slider("Meses a proyectar", options=[3, 6, 9, 12], value=6)
    graf = base_chart("Detección de períodos inusuales")
    graf.add_trace(go.Scatter(x=atipicos.index, y=atipicos.mensual, mode="lines",
                              name="Inflación mensual", line=dict(color="#1967ac", width=2)))
    sospechosos = atipicos[atipicos.atipico]
    graf.add_trace(go.Scatter(x=sospechosos.index, y=sospechosos.mensual, mode="markers",
                              name="Mes atípico", marker=dict(color="#ef644b", size=10,
                                                             symbol="circle-open", line=dict(width=2))))
    st.plotly_chart(graf, use_container_width=True)
    with st.spinner("Comparando modelos ARIMA y calculando pronósticos (la primera ejecución puede tardar)..."):
        resultado = modelos_cache(tuple(serie.values.tolist()), tuple(serie.index.astype(str)), meses)
    a, b = st.columns(2)
    a.metric("Mejor método en prueba", resultado["mejor"])
    b.metric("MAE (error absoluto medio)", fmt(resultado["mae"]) + " pp")
    fig = base_chart("Inflación observada y proyectada")
    fig.add_trace(go.Scatter(x=serie.tail(36).index, y=serie.tail(36).values,
                             mode="lines+markers", name="Observada",
                             line=dict(color="#147da2", width=3)))
    proj = resultado["pronostico"]
    fig.add_trace(go.Scatter(x=proj.fecha, y=proj.inflacion_proyectada, name="Pronóstico",
                             mode="lines+markers", line=dict(color="#ed9c46", width=3, dash="dash")))
    if "limite_inferior_80" in proj:
        fig.add_trace(go.Scatter(x=pd.concat([proj.fecha, proj.fecha[::-1]]),
                                 y=pd.concat([proj.limite_superior_80, proj.limite_inferior_80[::-1]]),
                                 fill="toself", fillcolor="rgba(235,156,70,.14)",
                                 line=dict(color="rgba(0,0,0,0)"), name="Intervalo 80 %"))
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Comparación de modelos** (últimos 12 meses como prueba)")
        st.dataframe(resultado["comparacion"].style.format({"MAE": "{:.3f}", "AIC": "{:.1f}", "BIC": "{:.1f}"}),
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Pronóstico mensual**")
        tabla_proy = proj.set_index("fecha").copy()
        tabla_proy.index = tabla_proy.index.strftime("%m/%Y")
        st.dataframe(tabla_proy.style.format("{:.2f}"), use_container_width=True)
    diagnostico = resultado.get("diagnostico")
    if diagnostico:
        st.caption(f"Diagnóstico Ljung–Box de residuos de {diagnostico['modelo']}: p = "
                   f"{diagnostico['p_ljung_box_12']:.3f} (12 rezagos). Un p > 0,05 no prueba independencia.")
    st.info("Un mes atípico no indica su causa. Los pronósticos extrapolan información pasada y pueden "
            "ser muy imprecisos ante shocks económicos. El intervalo solo está disponible cuando gana ARIMA.")
    st.download_button("⬇ Descargar pronósticos (CSV)", proj.to_csv(index=False).encode("utf-8-sig"),
                       file_name="pronosticos_inflacion.csv", mime="text/csv")

elif pagina == "💵 Poder adquisitivo":
    st.subheader("Simulador de poder adquisitivo")
    st.write("¿Cómo cambia el valor real de un sueldo o de un ahorro cuando aumenta el nivel de precios?")
    col1, col2 = st.columns(2)
    with col1:
        monto = st.number_input("Ingreso o ahorro inicial ($)", min_value=1000, max_value=1_000_000_000,
                                value=1_000_000, step=50_000)
        inflacion_elegida = st.slider("Inflación del período (%)", min_value=0.0, max_value=200.0,
                                     value=float(round(max(0, min(200, ultimo.interanual if pd.notna(ultimo.interanual) else 20)), 1)),
                                     step=0.5)
    with col2:
        aumento = st.slider("Aumento nominal de ingreso o ahorro (%)", min_value=0.0,
                            max_value=200.0, value=0.0, step=0.5)
        meses = st.slider("Distribuir el escenario en meses", min_value=1, max_value=24, value=12)
    resultado_sim = calcular_variacion_real(monto, inflacion_elegida, aumento)
    p, q, r = st.columns(3)
    p.metric("Valor nominal final", "$ " + fmt(resultado_sim["nominal"], 0))
    q.metric("Valor real en pesos iniciales", "$ " + fmt(resultado_sim["real"], 0))
    r.metric("Variación real", fmt(resultado_sim["variacion_real"]) + " %")
    n = np.arange(meses + 1)
    evolucion_inflacion = (1 + inflacion_elegida / 100) ** (n / meses)
    evolucion_salario = monto * (1 + aumento / 100) ** (n / meses)
    real = evolucion_salario / evolucion_inflacion
    fig = base_chart("Evolución nominal y poder de compra", "Pesos")
    fig.add_trace(go.Scatter(x=n, y=evolucion_salario, name="Monto nominal", line=dict(color="#1581b1", width=3)))
    fig.add_trace(go.Scatter(x=n, y=real, name="Poder adquisitivo real", line=dict(color="#e59d3d", width=3)))
    fig.update_xaxes(title="Mes del escenario")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Es una simulación hipotética con una trayectoria de crecimiento constante: "
               "el deslizador de inflación no representa un pronóstico de INDEC.")

elif pagina == "📝 Informe y descargas":
    st.subheader("Informe económico automático")
    with st.spinner("Preparando resultados..."):
        atipicos = anomalias_cache(tuple(serie.values.tolist()), tuple(serie.index.astype(str)))
        resultado = modelos_cache(tuple(serie.values.tolist()), tuple(serie.index.astype(str)), 6)
    informe = resumen(df, resultado, atipicos, conjunto.fuente)
    st.markdown("#### Interpretación reproducible")
    st.write(informe)
    st.caption("Este primer informe se genera con reglas estadísticas, no por un modelo de lenguaje.")
    st.divider()
    st.markdown("#### Redacción con Gemini (opcional)")
    st.write("Podés solicitar una reformulación con un modelo de lenguaje. "
             "La aplicación funciona aunque no tengas una clave de Gemini.")
    try:
        clave_guardada = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        clave_guardada = ""
    clave = st.text_input("API key de Gemini (si querés usarlo)", type="password",
                           help="No la publiques ni la incluyas en los archivos de GitHub.")
    clave = clave or clave_guardada
    if st.button("✨ Redactar análisis con IA", disabled=not bool(clave)):
        try:
            with st.spinner("Consultando Gemini (tiempo máximo de red: 20 segundos)..."):
                st.session_state["gemini_texto"] = obtener_ia(informe, clave)
            st.success("Texto generado. Revisá los números antes de usarlo.")
        except Exception as ex:
            st.error(f"Gemini no respondió: {str(ex)[:220]}. Podés utilizar el informe automático.")
    if st.session_state.get("gemini_texto"):
        st.markdown("**Resumen redactado por Gemini:**")
        st.write(st.session_state["gemini_texto"])
    st.divider()
    st.markdown("#### Guardar documentos")
    txt_elegido = st.session_state.get("gemini_texto") or informe
    st.download_button("⬇ Descargar informe (TXT)", txt_elegido.encode("utf-8-sig"),
                       file_name="informe_automatico.txt", mime="text/plain")
    try:
        contenido_pdf = pdf_resumen(df, conjunto, resultado, informe)
        st.download_button("⬇ Descargar informe (PDF)", contenido_pdf,
                           file_name="Informe_Inflacion_Argentina_IA.pdf", mime="application/pdf")
    except Exception as ex:
        st.warning(f"No se pudo preparar el PDF: {ex}")
    st.download_button("⬇ Descargar datos analizados (CSV)", df.to_csv().encode("utf-8-sig"),
                       file_name="ipc_analizado.csv", mime="text/csv")
    st.caption("El informe PDF contiene los indicadores y el análisis automático reproducible. "
               "La versión de Gemini, cuando se usa, se descarga aparte en TXT.")

st.divider()
st.caption("Proyecto académico · Python + Streamlit + Pandas + Plotly + Isolation Forest + ARIMA. "
           "Datos originados en INDEC mediante API de Series de Tiempo o espejo identificado.")

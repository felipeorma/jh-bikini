
# app_bikini_growth_streamlit.py
# Streamlit app: Proyección de crecimiento (tienda de bikinis - Chile)
# Autor: Felipe O.
# Ejecuta: streamlit run app_bikini_growth_streamlit.py

import os, io, math, textwrap
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =======================
# Utilidades
# =======================

SPANISH_MONTHS = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]
MONTH_TO_NUM = {m: i+1 for i, m in enumerate(SPANISH_MONTHS)}
NUM_TO_MONTH = {i+1: m for i, m in enumerate(SPANISH_MONTHS)}

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_venta_excel(file_or_path) -> pd.DataFrame:
    """
    Lee un Excel con hojas por año (2022..2026) y columnas típicas:
    Mes, Venta Histórica, Proyección Conservadora, Proyección Optimista, Venta Real, Variación, etc.
    Devuelve DF unificado: columns=[anio, mes_num, Mes, metric, valor]
    donde metric es 'Venta Histórica', 'Venta Real', etc.
    """
    if isinstance(file_or_path, str) and not os.path.exists(file_or_path):
        raise FileNotFoundError(f"No existe: {file_or_path}")
    xl = pd.ExcelFile(file_or_path)
    frames = []
    for sheet in xl.sheet_names:
        try:
            year = int(str(sheet).strip())
        except:
            # si hay hojas no-numéricas, se ignoran
            continue
        df = pd.read_excel(xl, sheet_name=sheet)
        df = _clean_cols(df)
        # normaliza columnas conocidas
        # keep only relevant columns that exist
        keep_cols = [c for c in df.columns if c in [
            "Mes","Venta Histórica","Proyección Conservadora","Proyección Optimista","Venta Real","Variación","Unidades","Uniddes","Unnamed: 2"
        ]]
        df = df[keep_cols].copy()
        if "Uniddes" in df.columns and "Unidades" not in df.columns:
            df.rename(columns={"Uniddes":"Unidades"}, inplace=True)
        if "Unnamed: 2" in df.columns and "Proyección Conservadora" not in df.columns:
            # algunos años tienen una columna vacía, la dejamos fuera
            df = df.drop(columns=["Unnamed: 2"])
        # agrega numeración del mes
        df["Mes"] = df["Mes"].astype(str).str.strip()
        df["mes_num"] = df["Mes"].map(MONTH_TO_NUM)
        df["anio"] = year
        frames.append(df)
    base = pd.concat(frames, ignore_index=True)
    return base

def month_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["anio","mes_num"]).reset_index(drop=True)

def build_seasonality(df_hist: pd.DataFrame, metric_col: str = "Venta Histórica") -> pd.DataFrame:
    """
    Calcula un índice de estacionalidad mensual (media de shares por año), normalizado a 12 (suma).
    Retorna DF con columnas: mes_num, Mes, seasonality_share
    """
    df = df_hist.dropna(subset=[metric_col]).copy()
    df["total_anual"] = df.groupby("anio")[metric_col].transform("sum")
    # evita división por cero
    df = df[df["total_anual"] > 0]
    df["share"] = df[metric_col] / df["total_anual"]
    seasonality = (df.groupby("mes_num")["share"].mean().reset_index()
                     .rename(columns={"share":"seasonality_share"}))
    # normaliza suma a 1.0
    total = seasonality["seasonality_share"].sum()
    if total > 0:
        seasonality["seasonality_share"] = seasonality["seasonality_share"] / total
    seasonality["Mes"] = seasonality["mes_num"].map(NUM_TO_MONTH)
    return seasonality[["mes_num","Mes","seasonality_share"]].sort_values("mes_num")

def project_next_year_monthly(
    last_year_df: pd.DataFrame,
    seasonality: pd.DataFrame,
    yoy_growth: float = 0.10,
) -> pd.DataFrame:
    """
    Proyecta N+1 usando total del último año * (1+yoy_growth) * seasonality_share
    last_year_df: filas del último año con Venta Histórica
    """
    base_total = last_year_df["Venta Histórica"].dropna().sum()
    target_total = base_total * (1.0 + yoy_growth)
    proj = seasonality.copy()
    proj["Proyección Base"] = proj["seasonality_share"] * target_total
    proj["anio"] = int(last_year_df["anio"].iloc[0]) + 1
    return proj[["anio","mes_num","Mes","Proyección Base"]]

def apply_new_stores(
    proj_df: pd.DataFrame,
    current_stores: int,
    new_stores: int,
    avg_mature_store_share: float = 1.0,
    ramp_first_month: float = 0.4,
    ramp_second_month: float = 0.6,
) -> pd.DataFrame:
    """
    Descompone la Proyección Base en:
      - Aporte tiendas actuales
      - Aporte nuevas tiendas (con ramp-up de 40% mes 1, 60% mes 2, 80% mes 3, 100% después)
    Asume que la Proyección Base ya incluye la suma de todas las tiendas (current).
    Para las nuevas, se agrega al total con la misma estacionalidad.
    """
    df = proj_df.copy()
    df["Existing Stores"] = df["Proyección Base"]
    # si no hay nuevas tiendas, retorna igual
    if new_stores <= 0:
        df["New Stores"] = 0.0
        df["Projected Revenue"] = df["Existing Stores"]
        return df

    # Creamos un factor de ramp-up por mes dentro del año [0..11]
    monthly_ramp = []
    for i in range(12):
        if i == 0:
            monthly_ramp.append(ramp_first_month)
        elif i == 1:
            monthly_ramp.append(ramp_second_month)
        elif i == 2:
            monthly_ramp.append(0.8)
        else:
            monthly_ramp.append(1.0)

    # Estimamos ingreso promedio por tienda madura como Proyección Base / current_stores
    # (aprox, sirve como punto de partida)
    per_store = df["Proyección Base"] / max(current_stores, 1)
    df["New Stores"] = per_store.values * new_stores * np.array(monthly_ramp) * avg_mature_store_share
    df["Projected Revenue"] = df["Existing Stores"] + df["New Stores"]
    return df

def add_low_season_uplift(df: pd.DataFrame, low_months: List[int], uplift_pct: float) -> pd.DataFrame:
    """Aplica un uplift (ej. promociones) en meses de baja temporada."""
    out = df.copy()
    mask = out["mes_num"].isin(low_months)
    out.loc[mask, "Projected Revenue"] *= (1.0 + uplift_pct)
    out.loc[mask, "New Stores"] *= (1.0 + uplift_pct)
    out.loc[mask, "Existing Stores"] *= (1.0 + uplift_pct)
    return out

def compute_break_even_and_units(
    df_proj: pd.DataFrame,
    price_per_unit: float,
    units_per_ticket: float,
    variable_cost_per_unit: float,
    fixed_cost_per_store: float,
    total_stores: int,
) -> pd.DataFrame:
    """
    Calcula:
      - ticket promedio = price_per_unit * units_per_ticket
      - margen unitario = price_per_unit - variable_cost_per_unit
      - margen % = margen unitario / price_per_unit
      - costo fijo total mensual = fixed_cost_per_store * total_stores
      - punto de equilibrio mensual (ingreso) = costo fijo / margen %
      - unidades a fabricar = Projected Revenue / price_per_unit
    """
    df = df_proj.copy()
    ticket_prom = price_per_unit * max(units_per_ticket, 0.0001)
    margin_unit = price_per_unit - variable_cost_per_unit
    margin_rate = margin_unit / max(price_per_unit, 0.0001)
    fixed_total = fixed_cost_per_store * total_stores
    be_rev = fixed_total / max(margin_rate, 0.0001)

    df["Ticket promedio (CLP)"] = ticket_prom
    df["Units Sold"] = df["Projected Revenue"] / max(price_per_unit, 0.0001)
    df["Gross Margin %"] = margin_rate * 100.0
    df["BreakEven Revenue"] = be_rev
    df["Above/Below BreakEven"] = df["Projected Revenue"] - be_rev

    return df, ticket_prom, margin_rate, be_rev

def estimate_turnover(df_proj: pd.DataFrame, avg_inventory_units: float) -> pd.DataFrame:
    """
    Calcula un turnover simplificado: unidades vendidas / inventario promedio.
    Si avg_inventory_units == 0, retorna NaN.
    """
    df = df_proj.copy()
    if avg_inventory_units and avg_inventory_units > 0:
        df["Inventory Turnover (x)"] = df["Units Sold"] / avg_inventory_units
    else:
        df["Inventory Turnover (x)"] = np.nan
    return df

# =======================
# App
# =======================

st.set_page_config(page_title="Proyección — Tienda de Bikinis (Chile)", layout="wide")

st.title("📈 Proyección de crecimiento — Tienda de Bikinis (Chile)")
st.caption("Modelo simple con estacionalidad + escenarios por nuevas tiendas, promociones y punto de equilibrio.")

with st.expander("ℹ️ Recomendaciones de datos para mayor precisión", expanded=False):
    st.markdown("""
**Para mejorar el modelo**, idealmente agrega:
- **Ventas por tienda** (si tienes más de una hoy), y por **canal** (online, tienda, pop‑up).
- **Costo variable por prenda** (COGS) y **gasto fijo mensual por tienda** (arriendo, sueldos, servicios).
- **Inventario promedio mensual** (unidades) y **stock de seguridad**.
- **Precio promedio por prenda** y/o **ítems por ticket** (para derivar ticket promedio).
- **Calendario de marketing** (promos, creators, ferias, pop-ups).
- **Aperturas planificadas** (fechas y ciudades).

Puedes seguir usando este archivo: **Venta.xlsx** (una hoja por año con columnas tipo “Mes, Venta Histórica, …”). Si subes un nuevo Excel con columnas parecidas, el app lo leerá. 
""")

# ======== Carga de datos
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("1) Datos")
    uploaded = st.file_uploader("Sube tu Excel (si no, intentamos leer 'Venta.xlsx' en el mismo folder)", type=["xlsx"])
    if uploaded is not None:
        base = load_venta_excel(uploaded)
    else:
        default_path = "Venta.xlsx"
        if os.path.exists(default_path):
            base = load_venta_excel(default_path)
        else:
            # intenta ruta del notebook si se ejecuta local
            alt_path = os.path.join(os.getcwd(), "Venta.xlsx")
            if os.path.exists(alt_path):
                base = load_venta_excel(alt_path)
            else:
                st.stop()
with col2:
    st.write("")
    st.write("")
    st.success("Datos cargados correctamente.")

base = base.dropna(subset=["Mes","mes_num","anio"]).copy()
base = base.replace({np.inf: np.nan, -np.inf: np.nan})

# Preview
st.dataframe(
    base.sort_values(["anio","mes_num"]).reset_index(drop=True),
    use_container_width=True, hide_index=True
)

# ======== Parámetros
st.subheader("2) Parámetros de proyección y negocio")

# Derivamos año más reciente con Venta Histórica
valid_hist = base.dropna(subset=["Venta Histórica"])
last_year = int(valid_hist["anio"].max()) if not valid_hist.empty else int(base["anio"].max())
seasonality = build_seasonality(valid_hist, metric_col="Venta Histórica")

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    yoy = c1.slider("Crecimiento YoY (tiendas actuales)", min_value=-0.5, max_value=1.0, value=0.15, step=0.01, help="Variación vs. total del último año.")
    current_stores = c2.number_input("Tiendas actuales", min_value=1, value=1, step=1)
    new_stores = c3.number_input("Nuevas tiendas a abrir (próximo año)", min_value=0, value=1, step=1)
    low_uplift = c4.slider("Promoción meses de baja (uplift %)", min_value=0, max_value=100, value=15, step=5)

with st.container():
    c5, c6, c7, c8 = st.columns(4)
    price_per_unit = c5.number_input("Precio promedio por prenda (CLP)", min_value=1000.0, value=24990.0, step=1000.0, format="%.0f")
    units_per_ticket = c6.number_input("Ítems por ticket (promedio)", min_value=0.1, value=1.4, step=0.1)
    var_cost_unit = c7.number_input("Costo variable por prenda (CLP)", min_value=500.0, value=9000.0, step=500.0, format="%.0f")
    fixed_cost_store = c8.number_input("Costo fijo mensual por tienda (CLP)", min_value=100000.0, value=4000000.0, step=100000.0, format="%.0f")

with st.container():
    c9, c10, c11 = st.columns(3)
    avg_inventory_units = c9.number_input("Inventario promedio mensual (unidades)", min_value=0.0, value=1200.0, step=50.0)
    ramp1 = c10.slider("Ramp-up tienda nueva — mes 1", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    ramp2 = c11.slider("Ramp-up tienda nueva — mes 2", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

low_season_months = st.multiselect(
    "Meses de baja temporada (para aplicar uplift/promos)",
    options=SPANISH_MONTHS, default=["Mayo","Junio","Julio","Agosto"]
)
low_month_nums = [MONTH_TO_NUM[m] for m in low_season_months]

# ======== Proyección
st.subheader("3) Proyección y escenarios")

last_year_df = (valid_hist[valid_hist["anio"]==last_year]
                .sort_values("mes_num")
                .reset_index(drop=True))

proj_base = project_next_year_monthly(last_year_df, seasonality, yoy_growth=yoy)
proj = apply_new_stores(proj_base, current_stores, new_stores, avg_mature_store_share=1.0,
                        ramp_first_month=ramp1, ramp_second_month=ramp2)
proj = add_low_season_uplift(proj, low_months=low_month_nums, uplift_pct=low_uplift/100.0)

# Cálculo de unidades, ticket y break-even
proj2, ticket_prom, margin_rate, be_rev = compute_break_even_and_units(
    proj, price_per_unit, units_per_ticket, var_cost_unit, fixed_cost_store, total_stores=(current_stores + new_stores)
)
proj2 = estimate_turnover(proj2, avg_inventory_units=avg_inventory_units)

# ======== KPIs
total_rev = proj2["Projected Revenue"].sum()
total_units = proj2["Units Sold"].sum()
months_below = (proj2["Above/Below BreakEven"] < 0).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ingresos proyectados (año +1)", f"${total_rev:,.0f} CLP")
k2.metric("Unidades a fabricar (año +1)", f"{total_units:,.0f}")
k3.metric("Ticket promedio", f"${ticket_prom:,.0f} CLP")
k4.metric("Meses bajo punto de equilibrio", f"{months_below}")

# ======== Gráficos
st.subheader("4) Visualizaciones")

# 4.1 Series históricas + proyección
hist_plot_df = valid_hist[["anio","mes_num","Mes","Venta Histórica"]].copy()
hist_plot_df["Tipo"] = "Histórico"
proj_plot_df = proj2[["anio","mes_num","Mes","Projected Revenue"]].copy()
proj_plot_df = proj_plot_df.rename(columns={"Projected Revenue":"Venta Histórica"})
proj_plot_df["Tipo"] = "Proyección"
plot_df = pd.concat([hist_plot_df, proj_plot_df], ignore_index=True)

fig_ts = px.line(
    plot_df, x="mes_num", y="Venta Histórica", color="anio",
    facet_row="Tipo", markers=True,
    category_orders={"mes_num": list(range(1,13))},
    labels={"mes_num":"Mes", "Venta Histórica":"CLP"}
)
fig_ts.update_xaxes(tickmode="array", tickvals=list(range(1,13)), ticktext=[NUM_TO_MONTH[i] for i in range(1,13)])
st.plotly_chart(fig_ts, use_container_width=True)

# 4.2 Aporte de nuevas tiendas vs actuales (barras apiladas)
stack_df = proj2[["Mes","Existing Stores","New Stores","Projected Revenue"]].copy()
fig_stack = go.Figure()
fig_stack.add_trace(go.Bar(name="Tiendas actuales", x=stack_df["Mes"], y=stack_df["Existing Stores"]))
fig_stack.add_trace(go.Bar(name="Nuevas tiendas", x=stack_df["Mes"], y=stack_df["New Stores"]))
fig_stack.update_layout(barmode="stack", yaxis_title="CLP")
st.plotly_chart(fig_stack, use_container_width=True)

# 4.3 Gap vs. punto de equilibrio
gap_df = proj2[["Mes","Projected Revenue","BreakEven Revenue","Above/Below BreakEven"]].copy()
fig_gap = go.Figure()
fig_gap.add_trace(go.Bar(name="Gap vs Break-even", x=gap_df["Mes"], y=gap_df["Above/Below BreakEven"]))
fig_gap.add_trace(go.Scatter(name="Break-even", x=gap_df["Mes"], y=gap_df["BreakEven Revenue"], mode="lines+markers"))
fig_gap.update_layout(yaxis_title="CLP")
st.plotly_chart(fig_gap, use_container_width=True)

# 4.4 Turnover estimado (si hay inventario promedio)
if not proj2["Inventory Turnover (x)"].isna().all():
    fig_to = px.bar(proj2, x="Mes", y="Inventory Turnover (x)", title="Rotación de productos (estimada)")
    st.plotly_chart(fig_to, use_container_width=True)

# ======== Tabla & descarga
st.subheader("5) Tabla de proyección y descarga")
show_cols = [
    "anio","Mes","Existing Stores","New Stores","Projected Revenue",
    "Ticket promedio (CLP)","Units Sold","Gross Margin %","BreakEven Revenue","Above/Below BreakEven","Inventory Turnover (x)"
]
table = proj2[show_cols].copy()
st.dataframe(table, use_container_width=True, hide_index=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV de proyección", data=csv, file_name="proyeccion_bikinis.csv", mime="text/csv")

# ======== Sugerencias para 'meses malos' (no quedar en negativo)
st.subheader("6) Tácticas para mejorar meses de baja")
st.markdown("""
- **Preventa con packs** (2 o 3 piezas) con envío gratis condicionado, enfocado en los meses de baja.
- **Bundles por look** (top + bottom + cover‑up) con precio ancla para subir el ticket sin bajar mucho margen.
- **Pop‑ups / ferias** en malls costeros y colaboraciones con gimnasios/boxes (tráfico invernal).
- **Creators y UGC**: micro‑influencers en Chile (códigos únicos por mes bajo → medir uplift).
- **Precios dinámicos**: descuentos *solo* en tallas/líneas lentas; evita canibalizar top sellers.
- **Email/SMS** con urgencia contextual** (clima caluroso, viajes, vacaciones de invierno).
- **Productos “all‑season”** (ropa de playa urbana, accesorios) para estabilizar la demanda.
""")

st.caption("Tip: Ajusta el **uplift %** de meses de baja y revisa si el gap contra el break-even desaparece.")

# ======== Notas
with st.expander("Notas del modelo"):
    st.markdown("""
- La estacionalidad se estima como el **promedio del share mensual** de ventas históricas por año.
- La proyección de N+1 parte del **total del último año × (1 + YoY)**, distribuido por estacionalidad.
- Las nuevas tiendas agregan ventas siguiendo un **ramp‑up** (40%, 60%, 80%, 100%). Puedes editar los factores.
- **Break‑even**: usa margen bruto y costos fijos mensuales por tienda para calcular la línea de equilibrio.
- **Unidades a fabricar** = ingreso proyectado / precio promedio por prenda.
- **Rotación** es aproximada: unidades vendidas / inventario promedio (si lo informas).
""")


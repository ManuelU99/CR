import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata
import re

# Streamlit config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Load main data
file_path = "data_bi_CR2.csv"
df = pd.read_csv(file_path)

# Define key columns
column_a = df.columns[0]   # Tipo_Acero_Limpio
column_b = df.columns[1]   # Grado_Acero
column_c = df.columns[2]   # Ciclo
column_d = df.columns[3]   # Familia
column_muestra_probeta_temp = df.columns[4]   # Muestra_Probeta_Temp
column_muestra = df.columns[5]  # Muestra
column_testtype = df.columns[6]  # Test type
column_index = df.columns[7]     # Muestra_Temp_TestType_Index
column_tipo_muestra = df.columns[8]  # Tipo de muestra (Sin °C)
column_soaking = df.columns[9]   # Soaking
column_temp_ensayo_req = "Temp Ensayo Req (merged)"
column_tipo_de_probeta = "Tipo de probeta"
column_op = "OP"
column_colada = "Colada.1"

columns_traccion = df.columns[10:23]
columns_dureza = df.columns[23:31]
columns_charpy = df.columns[31:42]

# 🔧 Format OP and Colada without decimals
df[column_op] = df[column_op].apply(lambda x: str(int(x)) if pd.notna(x) else x)
df[column_colada] = df[column_colada].apply(lambda x: str(int(x)) if pd.notna(x) else x)

# 🔧 Add 'Missing' option for OP filter
df['OP_display'] = df[column_op].fillna("Missing")

# Process Muestra and Temp
df['MuestraNum'] = df[column_muestra].astype(str)
df['Temp'] = pd.to_numeric(df[column_tipo_muestra], errors='coerce').round()

# Extract Group Number
df['GroupNumber'] = df[column_index].astype(str).apply(
    lambda x: int(re.findall(r'[TDC](\d+)', x)[0]) if re.findall(r'[TDC](\d+)', x) else 1
)

# Sidebar filters
st.sidebar.header("Filters")

all_familia = sorted(df[column_d].dropna().unique())
selected_familia = st.sidebar.multiselect("Select Familia", all_familia, default=all_familia)
df_filtered = df[df[column_d].isin(selected_familia)]

all_tipo = sorted(df_filtered[column_a].dropna().unique())
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", all_tipo, default=all_tipo)
df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]

all_ciclos = sorted(df_filtered[column_c].dropna().unique())
selected_ciclo = st.sidebar.multiselect("Select Ciclo", all_ciclos, default=all_ciclos)
df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]

all_colada = sorted(df_filtered[column_colada].dropna().unique())
selected_colada = st.sidebar.multiselect("Select Colada", all_colada, default=all_colada)
df_filtered = df_filtered[df_filtered[column_colada].isin(selected_colada)]

# 🔧 Use OP_display for filter
all_op = sorted(df_filtered['OP_display'].dropna().unique())
selected_op = st.sidebar.multiselect("Select OP", all_op, default=all_op)
df_filtered = df_filtered[df_filtered['OP_display'].isin(selected_op)]

all_tipo_de_muestra = sorted(df_filtered[column_tipo_de_probeta].dropna().unique())
selected_tipo_de_muestra = st.sidebar.multiselect("Select Tipo de probeta", all_tipo_de_muestra, default=all_tipo_de_muestra)
df_filtered = df_filtered[df_filtered[column_tipo_de_probeta].isin(selected_tipo_de_muestra)]

all_soaking = sorted(df_filtered[column_soaking].dropna().unique())
selected_soaking = st.sidebar.multiselect("Select Soaking", all_soaking, default=all_soaking)
df_filtered = df_filtered[df_filtered[column_soaking].isin(selected_soaking)]

unique_groups = sorted(df_filtered['GroupNumber'].unique())
selected_groups = st.sidebar.multiselect("Select Group Number", unique_groups, default=unique_groups)
df_filtered = df_filtered[df_filtered['GroupNumber'].isin(selected_groups)]

# Temp Ensayo Req filter
df_filtered[column_temp_ensayo_req] = df_filtered[column_temp_ensayo_req].astype(str)
all_temp_ensayo_req = sorted(df_filtered[column_temp_ensayo_req].dropna().unique())
selected_temp_ensayo_req = st.sidebar.multiselect("Select Temp Ensayo Req", all_temp_ensayo_req, default=all_temp_ensayo_req)
if selected_temp_ensayo_req:
    df_filtered = df_filtered[df_filtered[column_temp_ensayo_req].isin(selected_temp_ensayo_req)]

# Test type selection
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])
selected_columns = columns_traccion if test_type == "Traccion" else columns_dureza if test_type == "Dureza" else columns_charpy

# Muestra_Probeta_Temp filter
all_muestra_probeta = sorted(df_filtered[column_muestra_probeta_temp].dropna().unique())
selected_muestra_probeta = st.sidebar.multiselect("Select Muestra_Probeta_Temp", all_muestra_probeta, default=all_muestra_probeta)
df_filtered = df_filtered[df_filtered[column_muestra_probeta_temp].isin(selected_muestra_probeta)]

# Line toggle
show_lines = st.sidebar.checkbox("Show lines connecting dots", value=True)

# Quality control message
df_qc = pd.read_csv("https://raw.githubusercontent.com/ManuelU99/CR/refs/heads/main/Graph_Quality_Control_Check.csv")

reason_text = ""
if len(selected_tipo) == 1 and len(selected_ciclo) == 1 and len(selected_soaking) == 1:
    match = df_qc[
        (df_qc['Tipo_Acero_Limpio'].astype(str) == str(selected_tipo[0])) &
        (df_qc['Ciclo'].astype(str) == str(selected_ciclo[0])) &
        (df_qc['Soaking'].astype(str) == str(selected_soaking[0])) &
        (df_qc['TestType'].astype(str) == str(test_type))
    ]
    if not match.empty:
        reason_text = match.iloc[0]['Reason']

if reason_text:
    st.warning(f"⚠ Note for this graph: {reason_text}")

# Main graph
if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(
        id_vars=[column_a, column_b, column_c, column_d, column_muestra_probeta_temp,
                 column_muestra, column_testtype, column_index, column_tipo_muestra,
                 column_soaking, 'Temp', 'MuestraNum', 'GroupNumber',
                 column_temp_ensayo_req, column_tipo_de_probeta],
        value_vars=selected_columns,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    def normalize(text):
        return unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode('utf-8')

    def assign_color(m):
        m_norm = normalize(m)
        if "fluencia" in m_norm: return '#CC0066'
        if "rotura" in m_norm: return '#00009A'
        if "alarg" in m_norm: return '#009900'
        if "dureza" in m_norm and "ind" in m_norm and "max" in m_norm: return '#CC0066'
        if "dureza" in m_norm and "ind" in m_norm and "min" in m_norm: return '#EC36E0'
        if "dureza" in m_norm and "prom" in m_norm and "max" in m_norm: return '#00009A'
        if "dureza" in m_norm and "prom" in m_norm and "min" in m_norm: return '#1F7CC7'
        if "energ" in m_norm and "ind" in m_norm: return '#CC0066'
        if "energ" in m_norm and "prom" in m_norm: return '#00009A'
        if "area" in m_norm and "ind" in m_norm: return '#009900'
        if "area" in m_norm and "prom" in m_norm: return "#76E778"
        return '#999999'

    def assign_dash(m):
        m_norm = normalize(m)
        if "req" in m_norm and "max" in m_norm: return 'dash'
        if "req" in m_norm and "min" in m_norm: return 'dot'
        return 'solid'

    long_df['MeasurementClean'] = long_df['Measurement'].str.replace(r'\(merged\)', '', regex=True).str.strip()
    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['MeasurementClean'] + ' (Soaking ' + long_df[column_soaking].astype(str) + ')'
    long_df['IsPercentage'] = long_df['Measurement'].str.contains(r'\(%\)', regex=True)

    show_labels = len(long_df) <= 100

    fig = go.Figure()

    for (legend, color, dash, is_percentage), group in long_df.groupby(['Legend', 'ColorHex', 'LineDash', 'IsPercentage']):
        legend_norm = normalize(legend)
        show_text = group['Value'] if 'req' not in legend_norm and show_labels else None
        mode = 'lines+markers+text' if show_text is not None and show_lines else \
               'markers+text' if show_text is not None else \
               'lines+markers' if show_lines else 'markers'

        fig.add_trace(go.Scatter(
            x=group['Temp'],
            y=group['Value'],
            mode=mode,
            name=legend,
            line=dict(color=color, dash=dash),
            text=show_text,
            textposition='top center',
            yaxis='y2' if is_percentage else 'y',
            hovertemplate=f"<b>{legend}</b><br>Muestra: %{{customdata[0]}}<br>Temp: %{{x}}<br>Value: %{{y}}<extra></extra>",
            customdata=group[['MuestraNum']].values
        ))

    fig.update_layout(
        title=f"Dashboard - Curvas de Revenido: {test_type}",
        xaxis_title='Temp',
        yaxis=dict(title='Value'),
        yaxis2=dict(title='Value (%)', overlaying='y', side='right'),
        legend_title='Series',
        height=700,
        width=1200,
        hovermode='x'
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show filtered data table"):
        common_columns = [
            column_a, column_b, column_c, column_d, column_muestra_probeta_temp, column_muestra,
            column_testtype, column_index, column_tipo_muestra, column_soaking,
            column_temp_ensayo_req, column_tipo_de_probeta, column_op, column_colada
        ]
        # Use selected test type to get relevant measurement columns
        selected_columns_for_table = (
            columns_traccion if test_type == "Traccion"
            else columns_dureza if test_type == "Dureza"
            else columns_charpy
        )
        df_display = df_filtered[common_columns + list(selected_columns_for_table)].dropna(axis=1, how='all')
        st.write(df_display)


import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata

# Streamlit page config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
column_a = df.columns[0]  # Tipo_Acero_Limpio
column_b = df.columns[1]  # Grado_Acero
column_c = df.columns[2]  # Ciclo
column_d = df.columns[3]  # Familia
column_e = df.columns[4]  # Muestra_Probeta_Temp
column_f = df.columns[5]  # Tubo

columns_traccion = df.columns[7:20]
columns_dureza = df.columns[22:30]
columns_charpy = df.columns[33:44]

# Extract Muestra, Probeta, Temp
split_cols = df[column_e].str.split('-', expand=True)
df['Muestra'] = split_cols[0]
df['Probeta'] = split_cols[1]
df['Temp'] = split_cols[2].astype(float).round()

# Initialize selected filters (start with full DataFrame)
selected_familia = st.sidebar.multiselect("Select Familia", sorted(df[column_d].dropna().unique()))
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", sorted(df[column_a].dropna().unique()))
selected_ciclo = st.sidebar.multiselect("Select Ciclo", sorted(df[column_c].dropna().unique()))
selected_soaking = st.sidebar.multiselect("Select Soaking", sorted(df[column_f].dropna().unique()))

# Apply all filters at once (dynamic, bidirectional)
df_filtered = df.copy()
if selected_familia:
    df_filtered = df_filtered[df_filtered[column_d].isin(selected_familia)]
if selected_tipo:
    df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]
if selected_ciclo:
    df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]
if selected_soaking:
    df_filtered = df_filtered[df_filtered[column_f].isin(selected_soaking)]

# Update the available options AFTER filtering (so each dropdown updates dynamically)
available_familia = sorted(df[df[column_a].isin(df_filtered[column_a]) & df[column_c].isin(df_filtered[column_c]) & df[column_f].isin(df_filtered[column_f])][column_d].dropna().unique())
available_tipo = sorted(df[df[column_d].isin(df_filtered[column_d]) & df[column_c].isin(df_filtered[column_c]) & df[column_f].isin(df_filtered[column_f])][column_a].dropna().unique())
available_ciclo = sorted(df[df[column_d].isin(df_filtered[column_d]) & df[column_a].isin(df_filtered[column_a]) & df[column_f].isin(df_filtered[column_f])][column_c].dropna().unique())
available_soaking = sorted(df[df[column_d].isin(df_filtered[column_d]) & df[column_a].isin(df_filtered[column_a]) & df[column_c].isin(df_filtered[column_c])][column_f].dropna().unique())

# Reset selections if no data remains
if not available_familia:
    selected_familia = []
if not available_tipo:
    selected_tipo = []
if not available_ciclo:
    selected_ciclo = []
if not available_soaking:
    selected_soaking = []

# Select test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])
selected_columns = (
    columns_traccion if test_type == "Traccion"
    else columns_dureza if test_type == "Dureza"
    else columns_charpy
)

# Plotting
if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(
        id_vars=[column_a, column_c, column_d, column_e, column_f, 'Temp', 'Muestra', 'Probeta'],
        value_vars=selected_columns,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    def normalize(text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text

    def assign_color(m):
        m_norm = normalize(m)
        if "fluencia" in m_norm:
            return '#CC0066'
        if "rotura" in m_norm:
            return '#00009A'
        if "alarg" in m_norm:
            return '#009900'
        if "dureza" in m_norm and "ind" in m_norm and "max" in m_norm:
            return '#CC0066'
        if "dureza" in m_norm and "ind" in m_norm and "min" in m_norm:
            return '#EC36E0'
        if "dureza" in m_norm and "prom" in m_norm and "max" in m_norm:
            return '#00009A'
        if "dureza" in m_norm and "prom" in m_norm and "min" in m_norm:
            return '#1F7CC7'
        if "energ" in m_norm and "ind" in m_norm:
            return '#CC0066'
        if "energ" in m_norm and "prom" in m_norm:
            return '#00009A'
        if "area" in m_norm and "ind" in m_norm:
            return '#009900'
        if "area" in m_norm and "prom" in m_norm:
            return '#009900'
        return '#999999'

    def assign_dash(m):
        m_norm = normalize(m)
        has_req = "req" in m_norm
        has_max = "max" in m_norm or "máx" in m_norm
        has_min = "min" in m_norm or "mín" in m_norm
        if has_req and has_max:
            return 'dash'
        elif has_req and has_min:
            return 'dot'
        else:
            return 'solid'

    long_df['MeasurementClean'] = long_df['Measurement'].str.replace(r'\(merged\)', '', regex=True).str.strip()
    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['MeasurementClean'] + ' (Soaking ' + long_df[column_f].astype(str) + ')'

    fig = go.Figure()

    for (legend, color, dash), group in long_df.groupby(['Legend', 'ColorHex', 'LineDash']):
        legend_norm = normalize(legend)
        show_text = group['Value'] if 'req' not in legend_norm else None

        fig.add_trace(go.Scatter(
            x=group['Temp'],
            y=group['Value'],
            mode='lines+markers+text' if show_text is not None else 'lines+markers',
            name=legend,
            line=dict(color=color, dash=dash),
            text=show_text,
            textposition='top center',
            hovertemplate=(
                f"<b>{legend}</b><br>"
                f"Muestra: %{{customdata[0]}} / Probeta: %{{customdata[1]}}<br>"
                f"Temp: %{{x}}<br>"
                f"Value: %{{y}}<extra></extra>"
            ),
            customdata=group[['Muestra', 'Probeta']].values
        ))

    fig.update_layout(
        title=f"Dashboard - Curvas de Revenido: {test_type}",
        xaxis_title='Temp',
        yaxis_title='Value',
        xaxis=dict(tickangle=0),
        legend_title='Series',
        height=700,
        width=1200,
        hovermode='x'
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show filtered data table"):
        st.write(df_filtered)

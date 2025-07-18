import pandas as pd
import plotly.express as px
import streamlit as st
import unicodedata

# Set page config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_traccion = df.columns[7:20]    # H to T
columns_dureza = df.columns[22:30]     # W to AD
columns_charpy = df.columns[33:44]     # AH to AR

column_a = df.columns[0]               # Tipo_Acero_Limpio
column_c = df.columns[2]               # Ciclo
column_e = df.columns[4]               # Muestra_Probeta_Temp
column_f = df.columns[5]               # Tubo

# Extract Temp (number after second '-')
df['Temp'] = df[column_e].str.extract(r'-(?:[^-]*)-(\d+)').astype(float)

# Sidebar filters
st.sidebar.header("Filters")

# Smart filter 1: Tipo_Acero_Limpio
all_tipo = sorted(df[column_a].dropna().unique())
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", all_tipo, default=all_tipo)
df_filtered = df[df[column_a].isin(selected_tipo)]

# Smart filter 2: Ciclo
all_ciclos = sorted(df_filtered[column_c].dropna().unique())
selected_ciclo = st.sidebar.multiselect("Select Ciclo", all_ciclos, default=all_ciclos)
df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]

# Smart filter 3: Soaking
all_soaking = sorted(df_filtered[column_f].dropna().unique())
selected_soaking = st.sidebar.multiselect("Select Soaking", all_soaking, default=all_soaking)
df_filtered = df_filtered[df_filtered[column_f].isin(selected_soaking)]

# Select test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])

# Map test type to columns
if test_type == "Traccion":
    selected_columns = columns_traccion
elif test_type == "Dureza":
    selected_columns = columns_dureza
elif test_type == "Charpy":
    selected_columns = columns_charpy

if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(
        id_vars=[column_a, column_c, column_e, column_f, 'Temp'],
        value_vars=selected_columns,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    # Normalize text (lowercase, remove accents)
    def normalize(text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text

    # Assign color
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

    # Assign dash
    def assign_dash(m):
        m_norm = normalize(m)
        has_req = "req" in m_norm
        has_max = "max" in m_norm
        has_min = "min" in m_norm
        if has_req and has_max:
            return 'dash'
        elif has_req and has_min:
            return 'dot'
        else:
            return 'solid'


    # Clean Measurement (remove '(merged)' if present)
    long_df['MeasurementClean'] = long_df['Measurement'].str.replace(r'\(merged\)', '', regex=True).str.strip()
    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['MeasurementClean'] + ' (Soaking ' + long_df[column_f].astype(str) + ')'

    # Create color map per unique Legend
    color_discrete_map = dict(zip(long_df['Legend'], long_df['ColorHex']))

    # Plot
    fig = px.line(
        long_df,
        x='Temp',
        y='Value',
        color='Legend',
        line_dash='LineDash',
        color_discrete_map=color_discrete_map,
        markers=True,
        title=f"Dashboard - Curvas de Revenido: {test_type}",
        labels={'Temp': 'Temp', 'Value': 'Value'}
    )

    # Update layout: unified hover, custom tooltip
    fig.update_layout(
        xaxis=dict(tickangle=0),
        legend_title='Series',
        height=700,
        width=1200,
        hovermode='x unified'
    )

    # Custom hovertemplate: no LineDash, no extra trace info
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Temp=%{x}<br>Value=%{y}<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show filtered data table"):
        st.write(df_filtered)

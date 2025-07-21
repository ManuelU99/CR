import pandas as pd
import plotly.express as px
import streamlit as st
import unicodedata

# Set Streamlit page config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Load data
file_path = "data_bi_CR2.csv"
df = pd.read_csv(file_path)

# Define key columns
column_tipo = df.columns[0]               # Tipo_Acero_Limpio
column_grado = df.columns[1]             # Grado_Acero
column_ciclo = df.columns[2]             # Ciclo
column_familia = df.columns[3]           # Familia
column_muestra_probeta_temp = df.columns[4]  # Muestra_Probeta_Temp
column_muestra = df.columns[5]          # Muestra
column_testtype = df.columns[6]         # Test type
column_muestra_temp_index = df.columns[7]    # Muestra_Temp_TestType_Index
column_temp = df.columns[9]             # Tipo de muestra (Sin °C)
column_soaking = df.columns[9]          # Soaking
column_fullpath = df.columns[-1]        # Full File Path

# Identify column ranges
columns_traccion = df.columns[10:23]    # K to W
columns_dureza = df.columns[23:31]      # X to AE
columns_charpy = df.columns[31:43]      # AF to AQ

# Extract Temp
df['Temp'] = df[column_temp].astype(float).round()

# Extract group key from last digit after '-'
df['Group'] = df[column_muestra_temp_index].str.extract(r'-(\w\d+)$')

# Sidebar filters
st.sidebar.header("Filters")

all_familia = sorted(df[column_familia].dropna().unique())
selected_familia = st.sidebar.multiselect("Select Familia", all_familia, default=all_familia)
df_filtered = df[df[column_familia].isin(selected_familia)]

all_tipo = sorted(df_filtered[column_tipo].dropna().unique())
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", all_tipo, default=all_tipo)
df_filtered = df_filtered[df_filtered[column_tipo].isin(selected_tipo)]

all_ciclo = sorted(df_filtered[column_ciclo].dropna().unique())
selected_ciclo = st.sidebar.multiselect("Select Ciclo", all_ciclo, default=all_ciclo)
df_filtered = df_filtered[df_filtered[column_ciclo].isin(selected_ciclo)]

all_soaking = sorted(df_filtered[column_soaking].dropna().unique())
selected_soaking = st.sidebar.multiselect("Select Soaking", all_soaking, default=all_soaking)
df_filtered = df_filtered[df_filtered[column_soaking].isin(selected_soaking)]

all_group = sorted(df_filtered['Group'].dropna().unique())
selected_group = st.sidebar.multiselect("Select Group Number", all_group, default=all_group)
df_filtered = df_filtered[df_filtered['Group'].isin(selected_group)]

all_muestras = sorted(df_filtered[column_muestra].dropna().unique())
selected_muestras = st.sidebar.multiselect("Select Muestra", all_muestras, default=all_muestras)
df_filtered = df_filtered[df_filtered[column_muestra].isin(selected_muestras)]

# Select test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])

if test_type == "Traccion":
    selected_columns = columns_traccion
elif test_type == "Dureza":
    selected_columns = columns_dureza
else:
    selected_columns = columns_charpy

if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(
        id_vars=[column_tipo, column_ciclo, column_familia, column_muestra, 'Temp', column_soaking],
        value_vars=selected_columns,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    def normalize(text):
        text = str(text).lower()
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

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
        has_max = "max" in m_norm or "max" in m_norm
        has_min = "min" in m_norm or "min" in m_norm
        if has_req and has_max:
            return 'dash'
        elif has_req and has_min:
            return 'dot'
        else:
            return 'solid'

    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['MeasurementClean'] = long_df['Measurement'].str.replace(r'\(merged\)', '', regex=True).str.strip()
    long_df['LegendUnique'] = long_df['MeasurementClean'] + ' (Soaking ' + long_df[column_soaking].astype(str) + ')'
    long_df['YAxis'] = long_df['Measurement'].apply(lambda x: 'y2' if '(%)' in x else 'y1')

    fig = px.line(
        long_df,
        x='Temp',
        y='Value',
        color='LegendUnique',
        line_dash='LineDash',
        color_discrete_map=dict(zip(long_df['LegendUnique'], long_df['ColorHex'])),
        markers=True,
        title=f"Dashboard - Curvas de Revenido: {test_type}",
        labels={'Temp': 'Temp', 'Value': 'Value'}
    )

    for trace in fig.data:
        yaxis = 'y2' if any('(%)' in m for m in trace.name) else 'y1'
        trace.update(yaxis=yaxis)

    fig.update_layout(
        yaxis2=dict(overlaying='y', side='right', title='Value (%)'),
        height=700, width=1200, hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feedback section
    st.subheader("✅ Feedback")
    feedback = st.radio("Is this graph correct?", ["Yes", "No"])
    if feedback == "No":
        reason = st.text_input("Please describe why:")
        st.write(f"❌ Reason noted: {reason}")

    # Full File Path section
    st.subheader("📂 Full File Paths for Filtered Data")
    file_paths = df_filtered[[column_tipo, column_ciclo, column_fullpath]].drop_duplicates()
    for _, row in file_paths.iterrows():
        tipo = row[column_tipo]
        ciclo = row[column_ciclo]
        url = row[column_fullpath]
        st.markdown(f"🔗 **{tipo} - {ciclo}**: [Open File]({url})")

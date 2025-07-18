import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata

# Streamlit config
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

# Sidebar filters (simple cascade: Familia → rest)
st.sidebar.header("Filters")

# Step 1: Familia
all_familia = sorted(df[column_d].dropna().unique())
selected_familia = st.sidebar.multiselect("Select Familia", all_familia, default=all_familia)
df_filtered = df[df[column_d].isin(selected_familia)]

# Step 2: Tipo_Acero_Limpio (depends on Familia)
all_tipo = sorted(df_filtered[column_a].dropna().unique())
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", all_tipo, default=all_tipo)
df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]

# Step 3: Ciclo (depends on above)
all_ciclos = sorted(df_filtered[column_c].dropna().unique())
selected_ciclo = st.sidebar.multiselect("Select Ciclo", all_ciclos, default=all_ciclos)
df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]

# Step 4: Soaking (depends on above)
all_soaking = sorted(df_filtered[column_f].dropna().unique())
selected_soaking = st.sidebar.multiselect("Select Soaking", all_soaking, default=all_soaking)
df_filtered = df_filtered[df_filtered[column_f].isin(selected_soaking)]

# Step 5: Test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])
selected_columns = (
    columns_traccion if test_type == "Traccion"
    else columns_dureza if test_type == "Dureza"
    else columns_charpy
)

# Check and plot
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

    # Check if ≤100 points for data labels
    show_labels = len(long_df) <= 100

    fig = go.Figure()

    for (legend, color, dash), group in long_df.groupby(['Legend', 'ColorHex', 'LineDash']):
        legend_norm = normalize(legend)
        show_text = group['Value'] if ('req' not in legend_norm and show_labels) else None

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

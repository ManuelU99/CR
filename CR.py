import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata

# Set Streamlit page config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_traccion = df.columns[7:20]
columns_dureza = df.columns[22:30]
columns_charpy = df.columns[33:44]

column_a = df.columns[0]  # Tipo_Acero_Limpio
column_c = df.columns[2]  # Ciclo
column_e = df.columns[4]  # Muestra_Probeta_Temp
column_f = df.columns[5]  # Tubo

# Extract Muestra, middle ID, Temp
split_cols = df[column_e].str.split('-', expand=True)
df['Muestra'] = split_cols[0]
df['MiddleID'] = split_cols[1]
df['Temp'] = split_cols[2].astype(float).round()

# Sort to preserve input order
df['RowNumber'] = df.reset_index().index
df.sort_values(['Muestra', 'Temp', 'RowNumber'], inplace=True)

# Assign global group number:
# For each (Muestra, Temp), assign appearance index (1, 2, ...)
df['GroupInstance'] = df.groupby(['Muestra', 'Temp']).cumcount() + 1

# Now, determine the max number of group instances across all Muestra-Temp
max_group_number = df['GroupInstance'].max()

# Sidebar filters
st.sidebar.header("Filters")
all_tipo = sorted(df[column_a].dropna().unique())
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", all_tipo, default=all_tipo)
df_filtered = df[df[column_a].isin(selected_tipo)]

all_ciclos = sorted(df_filtered[column_c].dropna().unique())
selected_ciclo = st.sidebar.multiselect("Select Ciclo", all_ciclos, default=all_ciclos)
df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]

all_soaking = sorted(df_filtered[column_f].dropna().unique())
selected_soaking = st.sidebar.multiselect("Select Soaking", all_soaking, default=all_soaking)
df_filtered = df_filtered[df_filtered[column_f].isin(selected_soaking)]

# Provide selection: 1 to max_group_number
available_groups = list(range(1, max_group_number + 1))
selected_groups = st.sidebar.multiselect(
    "Select Group Number (global repetitions)", available_groups, default=available_groups
)

# Filter rows where GroupInstance is in selected groups
df_filtered = df_filtered[df_filtered['GroupInstance'].isin(selected_groups)]

# Select test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])

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
        id_vars=[column_a, column_c, column_e, column_f, 'Temp', 'Muestra', 'GroupInstance'],
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
                f"Muestra: %{{customdata[0]}} (GroupInstance %{{customdata[1]}})<br>"
                f"Temp: %{{x}}<br>"
                f"Value: %{{y}}<extra></extra>"
            ),
            customdata=group[['Muestra', 'GroupInstance']].values
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

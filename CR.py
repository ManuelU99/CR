import pandas as pd
import plotly.express as px
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="Dashboard Curvas de Revenido", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_traccion = df.columns[7:20]   # H to T
columns_dureza = df.columns[22:30]    # W to AD
columns_charpy = df.columns[33:44]    # AH to AR

column_a = df.columns[0]              # Tipo_Acero_Limpio
column_e = df.columns[4]              # Muestra_Probeta_Temp
column_f = df.columns[5]              # Tubo

# Extract Temp (number after second '-')
df['Temp'] = df[column_e].str.extract(r'-(?:[^-]*)-(\d+)').astype(float)

# Sidebar filters
st.sidebar.header("Filters")

selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", sorted(df[column_a].dropna().unique()), sorted(df[column_a].dropna().unique()))
filtered_df_for_soaking = df[df[column_a].isin(selected_tipo)] if selected_tipo else df
selected_soaking = st.sidebar.multiselect("Select Soaking", sorted(filtered_df_for_soaking[column_f].dropna().unique()), sorted(filtered_df_for_soaking[column_f].dropna().unique()))

# Select test type
test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])

# Map test type to columns
if test_type == "Traccion":
    selected_columns = columns_traccion
elif test_type == "Dureza":
    selected_columns = columns_dureza
elif test_type == "Charpy":
    selected_columns = columns_charpy

# Apply filters
df_filtered = df.copy()
if selected_tipo:
    df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]
if selected_soaking:
    df_filtered = df_filtered[df_filtered[column_f].isin(selected_soaking)]

if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(
        id_vars=[column_a, column_e, 'Temp', column_f],
        value_vars=selected_columns,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    # Assign ColorGroup and LineDash
    def assign_color_group(m):
        if "Fluencia" in m:
            return 'Fluencia'
        if "Rotura" in m:
            return 'Rotura'
        if "Alarg" in m:
            return 'Alarg'
        if "Dureza" in m and "Ind" in m and "Max" in m:
            return 'Dureza Ind Max'
        if "Dureza" in m and "Ind" in m and "Min" in m:
            return 'Dureza Ind Min'
        if "Dureza" in m and "Prom" in m and "Max" in m:
            return 'Dureza Prom Max'
        if "Dureza" in m and "Prom" in m and "Min" in m:
            return 'Dureza Prom Min'
        if "Energ" in m:
            return 'Energ'
        if "Area" in m:
            return 'Area'
        return 'Other'

    def assign_dash(m):
        if "Req" in m and "Max" in m:
            return 'dash'
        if "Req" in m and "Min" in m:
            return 'dot'
        else:
            return 'solid'

    long_df['ColorGroup'] = long_df['Measurement'].apply(assign_color_group)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['Measurement'] + ' (Soaking ' + long_df[column_f].astype(str) + ')'

    # Define color map per ColorGroup
    color_map = {
        'Fluencia': '#CC0066',
        'Rotura': '#00009A',
        'Alarg': '#009900',
        'Dureza Ind Max': '#CC0066',
        'Dureza Ind Min': '#EC36E0',
        'Dureza Prom Max': '#00009A',
        'Dureza Prom Min': '#1F7CC7',
        'Energ': '#CC0066',
        'Area': '#009900',
        'Other': '#999999'
    }

    fig = px.line(
        long_df,
        x='Temp',
        y='Value',
        color='ColorGroup',
        line_dash='LineDash',
        hover_name='Legend',
        markers=True,
        title=f"CR - {test_type}",
        labels={'Temp': 'Temp', 'Value': 'Value'}
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        coloraxis_showscale=False,
        legend_title='Series Group',
        height=700,
        width=1200
    )

    # Apply custom colors
    for trace in fig.data:
        group = trace.name
        trace.line.color = color_map.get(group, '#999999')

    # Show plot
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show data table
    if st.checkbox("Show filtered data table"):
        st.write(df_filtered)

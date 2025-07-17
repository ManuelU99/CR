import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(page_title="CR Dashboard 🚀", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_traccion = df.columns[7:20]    # H to T
columns_dureza = df.columns[22:30]     # W to AD
columns_charpy = df.columns[33:44]     # AH to AR

column_a = df.columns[0]               # Tipo_Acero_Limpio
column_e = df.columns[4]               # Muestra_Probeta_Temp
column_f = df.columns[5]               # Tubo

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

    # Assign color and dash style
    def assign_color(m):
        # Traccion group
        if "Fluencia" in m:
            return '#CC0066'
        if "Rotura" in m:
            return '#00009A'
        if "Alarg" in m:
            return '#009900'
        # Dureza group
        if "Dureza" in m and "Ind" in m and "Max" in m and "Req" not in m:
            return '#CC0066'
        if "Dureza" in m and "Ind" in m and "Min" in m and "Req" not in m:
            return '#EC36E0'
        if "Dureza" in m and "Ind" in m and "Max" in m and "Req" in m:
            return '#CC0066'
        if "Dureza" in m and "Ind" in m and "Min" in m and "Req" in m:
            return '#CC0066'
        if "Dureza" in m and "Prom" in m and "Max" in m and "Req" not in m:
            return '#00009A'
        if "Dureza" in m and "Prom" in m and "Min" in m and "Req" not in m:
            return '#1F7CC7'
        if "Dureza" in m and "Prom" in m and "Max" in m and "Req" in m:
            return '#00009A'
        if "Dureza" in m and "Prom" in m and "Min" in m and "Req" in m:
            return '#00009A'
        # Charpy group
        if "Energ" in m and "Ind" in m and "Min" in m:
            return '#CC0066'
        if "Energ" in m and "Prom" in m and "Min" in m:
            return '#00009A'
        if "Area" in m and "Ind" in m and "Min" in m:
            return '#009900'
        if "Area" in m and "Prom" in m and "Min" in m:
            return '#252423'
        # fallback
        return '#999999'

    def assign_dash(m):
        if "Req" in m and "Max" in m:
            return 'dash'
        if "Req" in m and "Min" in m:
            return 'dot'
        else:
            return 'solid'

    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['Measurement'] + ' (Soaking ' + long_df[column_f].astype(str) + ')'

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
        title=f"CR - {test_type}",
        labels={'Temp': 'Temp', 'Value': 'Value'}
    )

    fig.update_layout(
        xaxis=dict(tickangle=0),
        legend_title='Series',
        height=700,
        width=1200
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show filtered data table"):
        st.write(df_filtered)

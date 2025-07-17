import pandas as pd
import plotly.express as px
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="CR Tracción Dashboard 🚀", layout="wide")

# Load data
file_path = "data_bi_CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_ht = df.columns[7:20]  # H to T
column_a = df.columns[0]       # Tipo_Acero_Limpio
column_e = df.columns[4]       # Muestra_Probeta_Temp
column_f = df.columns[5]       # Tubo

# Extract Temp (number after second '-')
df['Temp'] = df[column_e].str.extract(r'-(?:[^-]*)-(\d+)').astype(float)

# Sidebar filters
st.sidebar.header("Filters")

selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", sorted(df[column_a].dropna().unique()), sorted(df[column_a].dropna().unique()))

# Dynamically get Tubo options based on selected Tipo_Acero_Limpio
filtered_df_for_tubos = df[df[column_a].isin(selected_tipo)] if selected_tipo else df
selected_tubo = st.sidebar.multiselect("Select Tubo", sorted(filtered_df_for_tubos[column_f].dropna().unique()), sorted(filtered_df_for_tubos[column_f].dropna().unique()))

# Apply filters
df_filtered = df.copy()

if selected_tipo:
    df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]

if selected_tubo:
    df_filtered = df_filtered[df_filtered[column_f].isin(selected_tubo)]

# Check if there's data left
if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    # Prepare long-format dataframe
    long_df = df_filtered.melt(
        id_vars=[column_a, column_e, 'Temp', column_f],
        value_vars=columns_ht,
        var_name='Measurement',
        value_name='Value'
    ).dropna(subset=['Value', 'Temp'])

    # Add line style and legend name
    long_df['LineStyle'] = long_df['Measurement'].apply(lambda x: 'dash' if 'Req' in x else 'solid')
    long_df['Legend'] = long_df['Measurement'] + ' (Tubo ' + long_df[column_f].astype(str) + ')'

    # Plot with Plotly
    fig = px.line(
        long_df,
        x='Temp',
        y='Value',
        color='Legend',
        line_dash='LineStyle',
        markers=True,
        title="CR - Tracción",
        labels={'Temp': 'Temp', 'Value': 'Value'}
    )

    fig.update_layout(
        xaxis=dict(tickangle=0),
        legend_title='Series',
        height=700,
        width=1200
    )

    # Show plot
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show filtered data table
    if st.checkbox("Show filtered data table"):
        st.write(df_filtered)

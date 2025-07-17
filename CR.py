import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="CR - Tracción Dashboard", layout="wide")

# Load data
file_path = r"C:\Users\60098360\Desktop\Excel files\Data bi CR.csv"
df = pd.read_csv(file_path)

# Define key columns
columns_ht = df.columns[7:20]  # H to T
column_a = df.columns[0]       # Tipo_Acero_Limpio
column_e = df.columns[4]       # Muestra_Probeta_Temp
column_f = df.columns[5]       # Tubo

# Extract Temp: number after second '-'
df['Temp'] = df[column_e].str.extract(r'-(?:[^-]*)-(\d+)').astype(float)

# Melt to long format
long_df = df.melt(
    id_vars=[column_a, column_e, 'Temp', column_f],
    value_vars=columns_ht,
    var_name='Measurement',
    value_name='Value'
).dropna(subset=['Value', 'Temp'])

# Add line style and legend name
long_df['LineStyle'] = long_df['Measurement'].apply(lambda x: 'dash' if 'Req' in x else 'solid')
long_df['Legend'] = long_df['Measurement'] + ' (Tubo ' + long_df[column_f].astype(str) + ')'

# Sidebar filters
st.sidebar.header("Filters")

# Tipo_Acero_Limpio filter
tipo_options = sorted(long_df[column_a].dropna().unique())
selected_tipos = st.sidebar.multiselect("Select Tipo_Acero_Limpio", tipo_options, default=tipo_options)

# Filter Tubo options dynamically
filtered_df = long_df[long_df[column_a].isin(selected_tipos)]
tubo_options = sorted(filtered_df[column_f].dropna().unique())
selected_tubos = st.sidebar.multiselect("Select Tubo", tubo_options, default=tubo_options)

# Apply both filters
final_df = filtered_df[filtered_df[column_f].isin(selected_tubos)]

# Plot
fig = px.line(
    final_df,
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

# Optional: Show data table
if st.checkbox("Show data table"):
    st.write(final_df)

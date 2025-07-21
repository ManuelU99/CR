import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata
import re
import os

# Streamlit config
st.set_page_config(page_title="Dashboard - Curvas de Revenido", layout="wide")

# Define local CSV storage path
local_csv_path = r"C:\Users\60098360\Desktop\Python codes\Graph Quality Control Check.xlsx"

# Load data
data_file_path = "data_bi_CR2.csv"
df = pd.read_csv(data_file_path)

# Define key columns
column_a = df.columns[0]
column_b = df.columns[1]
column_c = df.columns[2]
column_d = df.columns[3]
column_e = df.columns[4]
column_muestra = df.columns[5]
column_testtype = df.columns[6]
column_index = df.columns[7]
column_tipo_muestra = df.columns[8]
column_soaking = df.columns[9]
column_fullpath = df.columns[43]
columns_traccion = df.columns[10:23]
columns_dureza = df.columns[23:31]
columns_charpy = df.columns[31:42]

# Process Muestra and Temp
df['MuestraNum'] = df[column_muestra].astype(str)
df['Temp'] = pd.to_numeric(df[column_tipo_muestra], errors='coerce').round()
df['GroupNumber'] = df[column_index].astype(str).apply(lambda x: int(re.findall(r'[TDC](\d+)', x)[0]) if re.findall(r'[TDC](\d+)', x) else 1)

# Sidebar filters
st.sidebar.header("Filters")
selected_familia = st.sidebar.multiselect("Select Familia", sorted(df[column_d].dropna().unique()), default=sorted(df[column_d].dropna().unique()))
df_filtered = df[df[column_d].isin(selected_familia)]
selected_tipo = st.sidebar.multiselect("Select Tipo_Acero_Limpio", sorted(df_filtered[column_a].dropna().unique()), default=sorted(df_filtered[column_a].dropna().unique()))
df_filtered = df_filtered[df_filtered[column_a].isin(selected_tipo)]
selected_ciclo = st.sidebar.multiselect("Select Ciclo", sorted(df_filtered[column_c].dropna().unique()), default=sorted(df_filtered[column_c].dropna().unique()))
df_filtered = df_filtered[df_filtered[column_c].isin(selected_ciclo)]
selected_soaking = st.sidebar.multiselect("Select Soaking", sorted(df_filtered[column_soaking].dropna().unique()), default=sorted(df_filtered[column_soaking].dropna().unique()))
df_filtered = df_filtered[df_filtered[column_soaking].isin(selected_soaking)]
selected_muestras = st.sidebar.multiselect("Select Muestra", sorted(df_filtered['MuestraNum'].dropna().unique()), default=sorted(df_filtered['MuestraNum'].dropna().unique()))
df_filtered = df_filtered[df_filtered['MuestraNum'].isin(selected_muestras)]
selected_groups = st.sidebar.multiselect("Select Group Number", sorted(df_filtered['GroupNumber'].unique()), default=sorted(df_filtered['GroupNumber'].unique()))
df_filtered = df_filtered[df_filtered['GroupNumber'].isin(selected_groups)]

test_type = st.sidebar.selectbox("Select Test Type", ["Traccion", "Dureza", "Charpy"])
selected_columns = columns_traccion if test_type == "Traccion" else columns_dureza if test_type == "Dureza" else columns_charpy

if df_filtered.empty:
    st.warning("⚠ No data available for the selected filters.")
else:
    long_df = df_filtered.melt(id_vars=[column_a, column_b, column_c, column_d, column_e, column_muestra, column_testtype, column_index, column_tipo_muestra, column_soaking, 'Temp', 'MuestraNum', 'GroupNumber', column_fullpath], value_vars=selected_columns, var_name='Measurement', value_name='Value').dropna(subset=['Value', 'Temp'])

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
        if "area" in m_norm and "prom" in m_norm: return '#009900'
        return '#999999'

    def assign_dash(m):
        m_norm = normalize(m)
        if "req" in m_norm and ("max" in m_norm or "máx" in m_norm): return 'dash'
        if "req" in m_norm and ("min" in m_norm or "mín" in m_norm): return 'dot'
        return 'solid'

    long_df['MeasurementClean'] = long_df['Measurement'].str.replace(r'\\(merged\\)', '', regex=True).str.strip()
    long_df['ColorHex'] = long_df['Measurement'].apply(assign_color)
    long_df['LineDash'] = long_df['Measurement'].apply(assign_dash)
    long_df['Legend'] = long_df['MeasurementClean'] + ' (Soaking ' + long_df[column_soaking].astype(str) + ')'
    long_df['IsPercentage'] = long_df['Measurement'].str.contains(r'\\(%\\)', regex=True)

    show_labels = len(long_df) <= 100

    fig = go.Figure()
    for (legend, color, dash, is_percentage), group in long_df.groupby(['Legend', 'ColorHex', 'LineDash', 'IsPercentage']):
        legend_norm = normalize(legend)
        show_text = group['Value'] if ('req' not in legend_norm and show_labels) else None
        fig.add_trace(go.Scatter(x=group['Temp'], y=group['Value'], mode='lines+markers+text' if show_text is not None else 'lines+markers', name=legend, line=dict(color=color, dash=dash), text=show_text, textposition='top center', yaxis='y2' if is_percentage else 'y', hovertemplate=f"<b>{legend}</b><br>Muestra: %{{customdata[0]}}<br>Temp: %{{x}}<br>Value: %{{y}}<extra></extra>", customdata=group[['MuestraNum']].values))
    fig.update_layout(title=f"Dashboard - Curvas de Revenido: {test_type}", xaxis_title='Temp', yaxis=dict(title='Value'), yaxis2=dict(title='Value (%)', overlaying='y', side='right'), legend_title='Series', height=700, width=1200, hovermode='x')
    st.plotly_chart(fig, use_container_width=True)

    # Graph Quality Check interface storing locally
    if len(selected_tipo) == 1 and len(selected_ciclo) == 1:
        st.subheader("✅ Graph Quality Check")
        is_correct = st.radio("Is this graph correct?", ("Yes", "No"))
        reason = ""

        if is_correct == "No":
            reason = st.text_area("Please describe why the graph is incorrect:")
            if reason:
                entry = pd.DataFrame([{
                    "Familia": ",".join(selected_familia),
                    "Tipo_Acero_Limpio": selected_tipo[0],
                    "Ciclo": selected_ciclo[0],
                    "Soaking": ",".join(selected_soaking),
                    "GroupNumber": ",".join([str(g) for g in selected_groups]),
                    "TestType": test_type,
                    "IsCorrect": is_correct,
                    "Reason": reason
                }])
                try:
                    if os.path.exists(local_csv_path):
                        existing = pd.read_csv(local_csv_path)
                        pd.concat([existing, entry], ignore_index=True).to_csv(local_csv_path, index=False)
                    else:
                        entry.to_csv(local_csv_path, index=False)
                    st.success(f"✅ Feedback saved to: {local_csv_path}")
                except Exception as e:
                    st.error(f"❌ Error saving to file: {e}")
            else:
                st.warning("⚠ Please provide a reason for marking as incorrect.")
        else:
            st.success("✅ Marked as CORRECT!")

    # Show Full File Paths as clickable links
    st.subheader("📂 Full File Paths for Filtered Data")

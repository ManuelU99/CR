import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import f_oneway
import plotly.express as px
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import shapiro, ttest_ind, zscore
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import ast
from io import StringIO
from matplotlib.dates import date2num



#Option 1
def show_correlation_heatmaps(df):
    # Keep original selection and ADD column 18 (index 17)
    selected_columns = df.iloc[:, [0, 1, 2, 3, 4, 11, 12, 13, 14, 17]].copy()

    # Add mechanical properties and result column
    selected_columns["Sy [Mpa]"] = df["Sy [Mpa]"]
    selected_columns["Uts [MPa]"] = df["Uts [MPa]"]
    selected_columns["Test Result"] = df.iloc[:, 9]

    # Drop rows with missing values
    selected_columns.dropna(inplace=True)

    # Create figure and subplots using gridspec for equal size
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    # Define shared colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])  # [left, bottom, width, height]

    for i, (ax, result) in enumerate(zip(axes, ["OK", "ERROR"])):
        # Subset data by result type
        group = selected_columns[selected_columns["Test Result"] == result].drop("Test Result", axis=1)

        # Compute correlation matrix
        corr_matrix = group.corr()

        # Format annotation as percentages
        annot_matrix = corr_matrix.map(lambda x: f"{x * 100:.0f}%")

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=annot_matrix,
            fmt="",
            cmap="coolwarm",
            square=True,
            ax=ax,
            cbar=(i == 1),  # only add colorbar to second heatmap
            cbar_ax=cbar_ax if i == 1 else None,
            vmin=-1, vmax=1,
            annot_kws={'size': 9},
            linewidths=0.5,
            linecolor='white'
        )

        # Title and ticks
        ax.set_title(f"Correlation Matrix - Test Result: {result}", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)

        if i == 0:
            ax.tick_params(axis='y', labelsize=10)
        else:
            ax.set_yticklabels([])  # remove y labels for ERROR subplot

        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    # Set global title and adjust spacing
    plt.suptitle("Full Correlation Matrices by Test Result", fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 0.91, 1])  # reserve space for colorbar
    plt.show()

#Option 2
def show_yield_strength_by_quarter(df):
    df = df.copy()

    # Normalize and filter out "STILL TESTING"
    df["Test Result"] = df.iloc[:, 9].astype(str).str.strip().str.upper()
    df = df[df["Test Result"] != "STILL TESTING"]

    # Parse dates and drop rows with invalid/missing dates
    df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors='coerce')
    df = df.dropna(subset=["Fecha finalización"])  # ✅ avoid non-finite values for year/quarter

    # Create year/quarter labels
    df["Year"] = df["Fecha finalización"].dt.year.astype(int)
    df["Quarter"] = df["Fecha finalización"].dt.quarter.astype(int)
    df["Year-Quarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)

    # Prepare data for plotting
    df_plot = df[["Year-Quarter", "Sy [Mpa]", "Test Result"]].dropna()
    quarter_order = sorted(df_plot["Year-Quarter"].unique(), key=lambda x: pd.Period(x, freq='Q'))

    # Plot boxplot
    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(
        data=df_plot,
        x="Year-Quarter",
        y="Sy [Mpa]",
        hue="Test Result",
        order=quarter_order,
        palette={"OK": "green", "ERROR": "blue"}
    )

    # Dashed lines between quarters
    for i in range(len(quarter_order) - 1):
        ax.axvline(x=i + 0.5, linestyle='--', color='gray', alpha=0.5)

    # Add sample sizes
    grouped = df_plot.groupby(["Year-Quarter", "Test Result"])
    for (xval, result), group in grouped:
        x_idx = quarter_order.index(xval)
        hue_offset = -0.2 if result == "OK" else 0.2
        min_val = group["Sy [Mpa]"].min()
        ax.text(
            x_idx + hue_offset,
            min_val - 5,
            f"n={len(group)}",
            ha='center',
            va='top',
            fontsize=9,
            color='black',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='none', facecolor='white', alpha=0.6)
        )

    # Secondary Y-axis for error rate
    ax2 = ax.twinx()
    grouped_results = df_plot.groupby(["Year-Quarter", "Test Result"]).size().unstack(fill_value=0)
    error_rates = []
    x_positions = []
    for i, quarter in enumerate(quarter_order):
        if quarter == "2021-Q4":
            continue
        total = grouped_results.loc[quarter].sum()
        errors = grouped_results.loc[quarter].get("ERROR", 0)
        error_pct = (errors / total) * 100 if total > 0 else 0
        error_rates.append(error_pct)
        x_positions.append(i)

    # Plot error trendline
    ax2.plot(
        x_positions,
        error_rates,
        marker='o',
        linestyle='-',
        color='red',
        linewidth=2,
        label='Error Rate (%)'
    )

    for x, y in zip(x_positions, error_rates):
        ax2.text(x, y + 0.3, f"{int(round(y))}%", color='red', fontsize=9, ha='center')

    ax2.set_ylabel("Error Rate (%)", color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 35)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 15, ymax)

    ax.set_title("Yield Strength Distribution by Year and Quarter", fontsize=14)
    ax.set_xlabel("Year - Quarter", fontsize=12)
    ax.set_ylabel("Sy [Mpa]", fontsize=12)
    ax.set_xticks(range(len(quarter_order)))
    ax.set_xticklabels(quarter_order, rotation=0)
    ax.legend(title="Test Result for corrosion Method A")
    plt.tight_layout()
    plt.show()




# Option 3
def analyze_data_groups(df):
    df = df.copy()

    # Map relevant columns by index
    col_A = df.columns[0]   # Defl. [mm]
    col_B = df.columns[1]   # Ti [ºC]
    col_C = df.columns[2]   # Tf [ºC]
    col_D = df.columns[3]   # pHi
    col_E = df.columns[4]   # pHf
    col_L = df.columns[11]  # C (%)
    col_M = df.columns[12]  # Cr (%)
    col_N = df.columns[13]  # Mn (%)
    col_O = df.columns[14]  # Mo (%)
    col_P = df.columns[15]  # Sy [Mpa]
    col_Q = df.columns[16]  # Uts [MPa]
    col_R = df.columns[17]  # HRC
    col_result = df.columns[9]  # Test Result

    # Main targets to analyze against
    main_target_options = {
        "1": (col_P, "Sy [Mpa]"),
        "2": (col_Q, "Uts [MPa]"),
        "3": (col_R, "HRC")
    }

    print("Select the main variable to analyze errors against:")
    for key, (col, label) in main_target_options.items():
        print(f"{key} - {col} ({label})")
    main_choice = input("Enter 1, 2, or 3: ").strip()
    print()

    if main_choice not in main_target_options:
        print("Invalid choice for main variable.")
        return

    main_col, main_label = main_target_options[main_choice]
    # Set bin size for main variable
    if main_col == col_R:
        bin_size_main = 0.5 #Para HRC uso un rango mas chico
    else:
        bin_size_main = 5 #Para fluencia y rotura 

    # Secondary (cross) variable options
    secondary_options = {
        "1": (col_A, 0.2),  #Deflexion
        "2": (col_B, 0.5),  #Ti
        "3": (col_C, 0.5),  #Tf
        "4": (col_D, 0.05), #pHi
        "5": (col_E, 0.05), #pHf
        "6": (col_R, 0.5),  #HRC
        "7": (col_P, 5),    #Sy
        "8": (col_Q, 5)     #Uts
    }

    print("Select the variable to cross with:")
    for key, (col, _) in secondary_options.items():
        print(f"{key} - {col}")
    sec_choice = input("Enter a number from 1 to 8: ").strip()
    print()

    if sec_choice not in secondary_options:
        print("Invalid choice for secondary variable.")
        return

    compare_col, bin_size_sec = secondary_options[sec_choice]

    # Normalize data
    df["Test Result"] = df[col_result].astype(str).str.strip().str.upper()
    df[[compare_col, main_col]] = df[[compare_col, main_col]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[compare_col, main_col, "Test Result"])

    sec_min = df[compare_col].min()
    main_min = df[main_col].min()

    df["Sec_bin_start"] = (((df[compare_col] - sec_min + 1e-6) / bin_size_sec).apply(np.floor) * bin_size_sec) + sec_min
    df["Main_bin_start"] = (((df[main_col] - main_min + 1e-6) / bin_size_main).apply(np.floor) * bin_size_main) + main_min

    grouped = df.groupby(["Sec_bin_start", "Main_bin_start", "Test Result"]).size().unstack(fill_value=0)

    if "ERROR" not in grouped.columns or grouped["ERROR"].sum() == 0:
        print("No 'ERROR' test results found in the grouped data.")
        return

    max_error_idx = grouped["ERROR"].idxmax()
    max_error_count = grouped["ERROR"].max()

    sec_start = max_error_idx[0]
    sec_end = sec_start + bin_size_sec
    main_start = max_error_idx[1]
    main_end = main_start + bin_size_main

    print("Most ERROR results found in:")
    print(f"  Column '{compare_col}' range: {sec_start:.3f} – {sec_end:.3f}")
    print(f"  Column '{main_col}' ({main_label}) range: {main_start:.1f} – {main_end:.1f}")
    print(f"  ERROR count: {max_error_count}")

    top_n = grouped.sort_values("ERROR", ascending=False).head(5)
    print("\ntop 5 combinations with highest ERROR counts:")
    for idx, row in top_n.iterrows():
        s_start, m_start = idx
        s_end = s_start + bin_size_sec
        m_end = m_start + bin_size_main
        print(f"  {compare_col}: {s_start:.3f}-{s_end:.3f} | {main_col} ({main_label}): {m_start:.1f}-{m_end:.1f} → ERRORs: {row['ERROR']}")
# NOTA: Los bins mostrados en chat son [X,Y)


# Option 4: Time-graphs and Exploratory Analysis
def show_time_graph(df):
    df = df.copy()

    # Column references
    col_result = df.columns[9]         # "Test Result"
    col_date = df.columns[10]          # "Fecha finalización"
    col_P = df.columns[15]             # "Fuencia EUL Obt (MPa)"

    # Standardize and clean
    df["Test Result"] = df[col_result].astype(str).str.strip().str.upper()
    df["Fecha finalización"] = pd.to_datetime(df[col_date], format="%m/%d/%Y", errors='coerce')
    df = df.dropna(subset=["Fecha finalización", "Test Result"])

    # Create time groups
    df["Year"] = df["Fecha finalización"].dt.year
    df["Quarter"] = df["Fecha finalización"].dt.quarter
    df["Year-Quarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    quarter_order = sorted(df["Year-Quarter"].unique())

    while True:
        print("\nSelect a time-based analysis:")
        print("1 - Bar chart: Number of ERROR results by quarter (with error rate)")
        print("2 - Trend: Mean value of a variable by quarter (with error rate)")
        print("3 - Boxplot: Distribution of a variable by Test Result")

        subchoice = input("Enter 1 to 3: ").strip()
        print()

        if subchoice == "1":
            error_df = df[df["Test Result"] == "ERROR"]
            error_counts = error_df["Year-Quarter"].value_counts().reindex(quarter_order, fill_value=0)
            total_counts = df["Year-Quarter"].value_counts().reindex(quarter_order, fill_value=0)

            fig, ax1 = plt.subplots(figsize=(16, 6))
            ax1.bar(error_counts.index, error_counts.values, color='crimson')
            ax1.set_ylabel("Number of ERRORs", color='crimson', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='crimson')

            for i, val in enumerate(error_counts.values):
                ax1.text(i, val + 0.5, str(val), ha='center', va='bottom', fontsize=9)

            x_numeric = np.arange(len(error_counts))
            z = np.polyfit(x_numeric, error_counts.values, 1)
            p = np.poly1d(z)
            ax1.plot(error_counts.index, p(x_numeric), linestyle='--', color='black', label='Error Count Trend')

            # Compute error rate excluding 2021-Q4
            valid_quarters = [q for q in quarter_order if q != "2021-Q4"]
            error_rate_pct = (error_counts[valid_quarters] / total_counts[valid_quarters] * 100).fillna(0)
            x_positions = [quarter_order.index(q) for q in error_rate_pct.index]

            ax2 = ax1.twinx()
            ax2.plot(
                x_positions,
                error_rate_pct.values,
                marker='o',
                linestyle='-',
                color='red',
                linewidth=2,
                label='Error Rate (%)'
            )
            ax2.set_ylabel("Error Rate (%)", color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 30)
            ax2.invert_yaxis()

            for x, y in zip(x_positions, error_rate_pct.values):
                ax2.text(x, y + 0.5, f"{int(round(y))}%", ha='center', va='bottom', fontsize=9, color='red')

            plt.title("Number of ERROR Results and Error Rate by Quarter", fontsize=14)
            ax1.set_xlabel("Year - Quarter", fontsize=12)
            ax1.set_xticks(np.arange(len(quarter_order)))
            ax1.set_xticklabels(quarter_order, rotation=0)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

            plt.tight_layout()
            plt.show(block=False)

        elif subchoice == "2":
            print("Available numerical columns:")
            numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ["Year", "Quarter"]]
            for i, col in enumerate(numeric_cols):
                print(f"{i + 1} - {col}")
            idx = int(input("Select a variable to plot: ")) - 1
            col = numeric_cols[idx]

            color_palette = {"OK": "green", "ERROR": "#003366"}
            count_labels = df.groupby(["Year-Quarter", "Test Result"]).size().reset_index(name="Count")

            fig, ax1 = plt.subplots(figsize=(16, 6))
            sns.barplot(
                data=df,
                x="Year-Quarter",
                y=col,
                hue="Test Result",
                palette=color_palette,
                order=quarter_order,
                errorbar=None,
                ax=ax1
            )

            for container in ax1.containers:
                ax1.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

            for container in ax1.containers:
                for bar in container:
                    height = bar.get_height()
                    x = bar.get_x() + bar.get_width() / 2
                    test_result = bar.get_label()
                    xtick_index = int(round(x))
                    if 0 <= xtick_index < len(ax1.get_xticks()):
                        try:
                            yq_label = ax1.get_xticklabels()[xtick_index].get_text()
                            count = count_labels[
                                (count_labels["Year-Quarter"] == yq_label) &
                                (count_labels["Test Result"] == test_result)
                            ]["Count"].values
                            if len(count) > 0:
                                ax1.text(
                                    x,
                                    height - (0.05 * height),
                                    f"n={count[0]}",
                                    ha="center",
                                    va="top",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="white"
                                )
                        except:
                            pass

            for label in ["OK", "ERROR"]:
                temp = df[df["Test Result"] == label]
                grouped = temp.groupby("Year-Quarter")[col].mean().reindex(quarter_order)
                if grouped.notna().sum() >= 2:
                    x_vals = np.arange(len(quarter_order))
                    y_vals = grouped.values
                    mask = ~np.isnan(y_vals)
                    z = np.polyfit(x_vals[mask], y_vals[mask], 1)
                    p = np.poly1d(z)
                    ax1.plot(grouped.index, p(x_vals), linestyle='--', color=color_palette[label], label=f"{label} Trendline")

            # Error rate line
            error_counts = df[df["Test Result"] == "ERROR"]["Year-Quarter"].value_counts().reindex(quarter_order, fill_value=0)
            total_counts = df["Year-Quarter"].value_counts().reindex(quarter_order, fill_value=0)
            valid_quarters = [q for q in quarter_order if q != "2021-Q4"]
            error_rate_pct = (error_counts[valid_quarters] / total_counts[valid_quarters] * 100).fillna(0)
            x_positions = [quarter_order.index(q) for q in error_rate_pct.index]

            ax2 = ax1.twinx()
            ax2.plot(
                x_positions,
                error_rate_pct.values,
                marker='o',
                linestyle='-',
                color='red',
                linewidth=2,
                label='Error Rate (%)'
            )
            ax2.set_ylabel("Error Rate (%)", color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 30)
            ax2.invert_yaxis()

            for x, y in zip(x_positions, error_rate_pct.values):
                ax2.text(x, y + 0.5, f"{int(round(y))}%", ha='center', va='bottom', fontsize=9, color='red')

            ax1.set_title(f"Average {col} by Quarter and Test Result (with Error Rate)", fontsize=14)
            ax1.set_xlabel("Year - Quarter", fontsize=12)
            ax1.set_ylabel(f"Average {col}", fontsize=12)
            ax1.set_xticks(np.arange(len(quarter_order)))
            ax1.set_xticklabels(quarter_order, rotation=45)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

            plt.tight_layout()
            plt.show(block=False)

        elif subchoice == "3":
            print("Available numerical columns:")
            numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ["Year", "Quarter"]]
            for i, col in enumerate(numeric_cols):
                print(f"{i + 1} - {col}")
            idx = int(input("Select a variable to compare by Test Result: ")) - 1
            col = numeric_cols[idx]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Test Result", y=col)
            plt.title(f"Distribution of {col} by Test Result")
            plt.tight_layout()
            plt.show(block=False)




# Option 5: Average of selected columns by year and quarter
def plot_avg_by_quarter(df):
    df = df.copy()

    # Column references (by position)
    col_date = df.columns[10]     # Column K: "Fecha finalización"
    col_result = df.columns[9]    # Column J: "Resultado"
    col_hours = df.columns[7]     # Column H: "Horas Progreso"

    # Variables (label → column reference)
    variables = {
        "Defl. [mm]": df.columns[0],
        "Ti [ºC]": df.columns[1],
        "Tf [ºC]": df.columns[2],
        "pHi": df.columns[3],
        "pHf": df.columns[4],
        "C (%)": df.columns[11], #L
        "Cr (%)": df.columns[12],
        "Mn (%)": df.columns[13],
        "Mo (%)": df.columns[14],
        "S (%)": df.columns[15],
        "P (%)": df.columns[16],
        "Sy [MPa]": df.columns[17],
        "Uts [MPa]": df.columns[18],
        "HRC": df.columns[19],
    }

    subfilters = ["ALL", "OK", "ERROR", "<=100h", "100<x<720h"]

    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df = df.dropna(subset=[col_date, col_result, col_hours])
    df[col_result] = df[col_result].astype(str).str.strip().str.upper()
    df["Year-Quarter"] = df[col_date].dt.to_period("Q").astype(str)

    def apply_subfilter(data, subfilter):
        if subfilter == "OK":
            return data[data[col_result] == "OK"]
        elif subfilter == "ERROR":
            return data[data[col_result] == "ERROR"]
        elif subfilter == "<=100h":
            return data[data[col_hours] <= 100]
        elif subfilter == "100<x<720h":
            return data[(data[col_hours] > 100) & (data[col_hours] < 720)]
        return data

    fig = go.Figure()
    buttons = []
    trace_idx = 0
    total_combos = len(variables) * len(subfilters)

    for var_label, var_col in variables.items():
        for sub in subfilters:
            df_filtered = apply_subfilter(df, sub).copy()
            df_filtered = df_filtered.dropna(subset=[var_col])  # Skip NaNs

            if df_filtered.empty:
                continue

            avg_per_q = df_filtered.groupby("Year-Quarter")[var_col].mean().reset_index(name="avg")
            filtered_count = df_filtered.groupby("Year-Quarter")[var_col].count().reset_index(name="n_filtered")
            total_count = df.groupby("Year-Quarter")[var_col].count().reset_index(name="n_total")

            grouped = avg_per_q.merge(filtered_count, on="Year-Quarter", how="left")
            grouped = grouped.merge(total_count, on="Year-Quarter", how="left")
            grouped["text"] = grouped.apply(lambda row: f"n={int(row['n_filtered'])}/{int(row['n_total'])}", axis=1)
            grouped = grouped.sort_values("Year-Quarter")

            fig.add_trace(go.Scatter(
                x=grouped["Year-Quarter"],
                y=grouped["avg"],
                mode="lines+markers+text",
                text=grouped["text"],
                textposition="top center",
                name=f"{var_label} | {sub}",
                visible=False
            ))

            if sub == "<=100h" and len(grouped) > 1 and not grouped["avg"].isnull().all():
                x_vals = np.arange(len(grouped))
                y_vals = grouped["avg"].values
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                trend = slope * x_vals + intercept
                fig.add_trace(go.Scatter(
                    x=grouped["Year-Quarter"],
                    y=trend,
                    mode="lines",
                    line=dict(dash="dash", width=1),
                    name=f"Trend | {var_label} | {sub}",
                    visible=False,
                    showlegend=False
                ))
                include_trend = True
            else:
                fig.add_trace(go.Scatter(
                    x=grouped["Year-Quarter"],
                    y=[None] * len(grouped),
                    mode="lines",
                    line=dict(dash="dash", width=1),
                    name=f"Trend | {var_label} | {sub}",
                    visible=False,
                    showlegend=False
                ))

            visibility = [False] * (total_combos * 2)
            visibility[trace_idx * 2] = True
            visibility[trace_idx * 2 + 1] = True

            buttons.append({
                "label": f"{var_label} | {sub}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "yaxis": {"title": f"Average of {var_label}"},
                        "title": f"{var_label} over Time — {sub}"
                    }
                ]
            })

            trace_idx += 1

    if fig.data:
        fig.data[0].visible = True
        fig.data[1].visible = True

    fig.update_layout(
        title="Variable Trends by Filter (Quarter-Year)",
        xaxis_title="Year-Quarter",
        yaxis_title="Average Value",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "xanchor": "center",
            "y": 1.2,
            "yanchor": "top"
        }]
    )

    fig.show()



#Option 6-1: PCA graph
def show_pca_scatter(df):
    df = df.copy()

    # Define column names
    test_result_col = "Resultado"
    pca_columns = [
        "Defl. [mm]",
        "Ti [ºC]",
        "Tf [ºC]",
        "pHi",
        "pHf",
        "C (%)",
        "Cr (%)",
        "Mo (%)",
        "Mn (%)",
        "Sy [Mpa]",
        "Uts [MPa]",
        "HRC"
    ]

    # Clean test result
    df[test_result_col] = df[test_result_col].astype(str).str.strip().str.upper()

    # Drop missing values
    df = df.dropna(subset=pca_columns + [test_result_col])
    if df.shape[0] < 2:
        print("Not enough valid rows for PCA scatter.")
        return

    # Standardize and apply PCA
    X = df[pca_columns].apply(pd.to_numeric, errors='coerce').values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    # Prepare plot data
    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_pca["Resultado"] = df[test_result_col].values

    # Color map
    color_map = {"OK": "green", "ERROR": "red"}

    # Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Resultado",
        palette=color_map,
        s=60,
        alpha=0.8
    )
    plt.title("PCA Projection Colored by Test Result")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Option 6-2: PCA graph
def plot_pca_loadings(df):
    df = df.copy()

    pca_columns = [
        "Defl. [mm]",
        "Ti [ºC]",
        "Tf [ºC]",
        "pHi",
        "pHf",
        "C (%)",
        "Cr (%)",
        "Mo (%)",
        "Mn (%)",
        "Sy [Mpa]",
        "Uts [MPa]",
        "HRC"
    ]

    df = df.dropna(subset=pca_columns)
    if df.shape[0] < 2:
        print("Not enough valid rows to compute loadings.")
        return

    X = df[pca_columns].apply(pd.to_numeric, errors="coerce").values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=pca_columns)

    # Plot
    plt.figure(figsize=(12, 6))
    loading_df.plot(kind="bar", figsize=(12, 6), color=["green", "#003366"])
    plt.title("PCA Loadings: Variable Contribution to PC1 and PC2")
    plt.xlabel("Variables")
    plt.ylabel("Loading Value")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


#Option 7: Statistical analysis
def compare_ok_vs_error(df):
    df = df.copy()

    # Ensure proper data types
    df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
    df["Horas Progreso"] = pd.to_numeric(df["Horas Progreso"], errors='coerce')
    df["pHf"] = pd.to_numeric(df["pHf"], errors='coerce')
    df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors='coerce')

    # Remove invalid dates
    df.dropna(subset=["Fecha finalización"], inplace=True)

    # Create Year-Quarter label
    df["Year-Quarter"] = df["Fecha finalización"].dt.to_period("Q").astype(str)

    # Filter to OK and ERROR results
    df_filtered = df[df["Resultado"].isin(["OK", "ERROR"])]

    # ---------------- TABLE 1: OK vs ERROR ----------------
    variables = [
        "Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf",
        "C (%)", "Cr (%)", "Mo (%)", "Mn (%)",
        "Sy [Mpa]", "Uts [MPa]", "HRC"
    ]

    table1 = []
    headers1 = ["Variable", "OK Mean ± Std", "ERROR Mean ± Std", "p-value"]

    for col in variables:
        subset = df_filtered[[col, "Resultado"]].dropna()
        ok_vals = pd.to_numeric(subset[subset["Resultado"] == "OK"][col], errors='coerce').dropna()
        err_vals = pd.to_numeric(subset[subset["Resultado"] == "ERROR"][col], errors='coerce').dropna()

        if len(ok_vals) >= 2 and len(err_vals) >= 2:
            t_stat, p_val = ttest_ind(ok_vals, err_vals, equal_var=False)
            ok_str = f"{ok_vals.mean():.2f} ± {ok_vals.std():.2f} (n={len(ok_vals)})"
            err_str = f"{err_vals.mean():.2f} ± {err_vals.std():.2f} (n={len(err_vals)})"
            table1.append([col, ok_str, err_str, f"{p_val:.4f}"])
        else:
            table1.append([col, "N/A", "N/A", "N/A"])

    print("\nNOTE: A p-value < 0.05 typically indicates a statistically significant difference between OK and ERROR groups.")
    print("\nStatistical Comparison: OK vs ERROR\n")
    print(tabulate(table1, headers=headers1, tablefmt="grid"))

    # ---------------- TABLE 2: Duration < 100 hs ----------------
    table2 = []
    headers2 = ["Variable", "Mean ± Std (<100hs)", "n", "p-value vs rest"]

    short_df = df[df["Horas Progreso"] < 100]
    rest_df = df[df["Horas Progreso"] >= 100]

    for col in ["Defl. [mm]", "pHf"]:
        short_vals = pd.to_numeric(short_df[col], errors='coerce').dropna()
        rest_vals = pd.to_numeric(rest_df[col], errors='coerce').dropna()

        if len(short_vals) >= 2 and len(rest_vals) >= 2:
            t_stat, p_val = ttest_ind(short_vals, rest_vals, equal_var=False)
            avg_std = f"{short_vals.mean():.2f} ± {short_vals.std():.2f}"
            table2.append([col, avg_std, len(short_vals), f"{p_val:.4f}"])
        else:
            table2.append([col, "N/A", "N/A", "N/A"])

    print("\nSubset Analysis: Tests with duration < 100 hours\n")
    print(tabulate(table2, headers=headers2, tablefmt="grid"))

    # ---------------- GRAPH: Bar Chart + Trendlines ----------------
    # Group 1: Duration < 100 hs
    short_avg = short_df.groupby("Year-Quarter")["pHf"].mean()

    # Group 2: Result == OK
    ok_avg = df[df["Resultado"] == "OK"].groupby("Year-Quarter")["pHf"].mean()

    # Sorted quarter-year labels
    all_quarters = sorted(set(short_avg.index).union(ok_avg.index), key=lambda q: pd.Period(q, freq='Q'))
    x = np.arange(len(all_quarters))

    short_vals = np.array([short_avg.get(q, np.nan) for q in all_quarters])
    ok_vals = np.array([ok_avg.get(q, np.nan) for q in all_quarters])

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35

    # Bars
    ax.bar(x - width/2, short_vals, width, label='Dur < 100 hs', color='blue', alpha=0.7)
    ax.bar(x + width/2, ok_vals, width, label='Result == OK', color='green', alpha=0.7)

    # Trendlines
    def add_trendline(x, y, color, label, linestyle):
        mask = ~np.isnan(y)
        if sum(mask) >= 3:
            coefs = Polynomial.fit(x[mask], y[mask], deg=1).convert().coef
            trend = Polynomial(coefs)(x)
            ax.plot(x, trend, color=color, linestyle=linestyle, linewidth=2, label=f"{label} Trend")

    add_trendline(x, short_vals, color='blue', label='Dur < 100 hs', linestyle='--')
    add_trendline(x, ok_vals, color='green', label='Result == OK', linestyle='--')

    # Axes and labels
    ax.set_xlabel('Quarter-Year')
    ax.set_ylabel('Average pHf')
    ax.set_title('Average pHf per Quarter-Year (Bars + Trendlines)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_quarters, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



#Option 8: Decision-Tree
def build_decision_tree(df):
    df = df.copy()

    # Define target and features
    target_col = "Resultado"
    feature_cols = [
        "Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf",
        "C (%)", "Cr (%)", "Mo (%)", "Mn (%)",
        "Sy [Mpa]", "Uts [MPa]", "HRC"
    ]

    # Clean and encode
    df[target_col] = df[target_col].astype(str).str.strip().str.upper()
    df = df[df[target_col].isin(["OK", "ERROR"])]
    df = df.dropna(subset=feature_cols + [target_col])

    # Encode target: OK=0, ERROR=1
    df["Target"] = (df[target_col] == "ERROR").astype(int)

    X = df[feature_cols]
    y = df["Target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    # Plot tree
    plt.figure(figsize=(18, 8))
    plot_tree(clf, feature_names=feature_cols, class_names=["OK", "ERROR"], filled=True)
    plt.title("Decision Tree Classifier (Depth=3)")
    plt.show()

    return {
        "classification_report": pd.DataFrame(report).transpose(),
        "confusion_matrix": matrix
    }

# Option 9: 
def analyze_responsable_influence(df):
    df = df.copy()

    # Standardize columns
    df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
    df["Responsable Inicio"] = df["Responsable Inicio"].astype(str).str.strip()
    df["Resultado_bin"] = df["Resultado"].eq("ERROR").astype(int)

    # Menu
    print("\nSelect an analysis option:")
    print("1 - Bar chart: % of total ERRORs per Responsable")
    print("2 - Variable deviations (OK vs ERROR) per Responsable")
    print("3 - Error trend over time per Responsable")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        # Count ERRORs and total per responsable
        error_counts = df[df["Resultado"] == "ERROR"].groupby("Responsable Inicio").size()
        total_errors = error_counts.sum()
        total_counts = df.groupby("Responsable Inicio").size()

        error_share = (error_counts / total_errors).fillna(0) * 100

        chart_df = pd.DataFrame({
            "% of Total ERRORs": error_share,
            "ERROR Count": error_counts,
            "Total Tests": total_counts
        }).fillna(0).reset_index()

        chart_df["Label"] = chart_df["ERROR Count"].astype(int).astype(str) + "/" + chart_df["Total Tests"].astype(int).astype(str)

        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=chart_df, x="Responsable Inicio", y="% of Total ERRORs", color='green')
        plt.xticks(rotation=45, ha="right")
        plt.title("% of Total ERRORs by Lab Responsible")
        plt.ylabel("% of All ERRORs")
        plt.xlabel("Responsable Inicio")

        # Add text labels inside bars
        for i, row in chart_df.iterrows():
            ax.text(i, row["% of Total ERRORs"] + 0.5, row["Label"], ha="center", va="bottom", fontsize=9, color="black")

        plt.tight_layout()
        plt.show()

    elif choice == "2":
        # Variables to analyze
        variables = ["Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf"]
        result = []

        for responsable in df["Responsable Inicio"].unique():
            df_res = df[df["Responsable Inicio"] == responsable]
            if df_res.empty:
                continue

            for var in variables:
                if var in df.columns:
                    ok_vals = df_res[df_res["Resultado"] == "OK"][var].dropna()
                    err_vals = df_res[df_res["Resultado"] == "ERROR"][var].dropna()

                    if len(ok_vals) >= 2 and len(err_vals) >= 2:
                        diff = err_vals.mean() - ok_vals.mean()
                        result.append({
                            "Responsable": responsable,
                            "Variable": var,
                            "OK Mean": round(ok_vals.mean(), 2),
                            "ERROR Mean": round(err_vals.mean(), 2),
                            "Delta (ERROR - OK)": round(diff, 2)
                        })

        result_df = pd.DataFrame(result)
        print(tabulate(result_df, headers="keys", tablefmt="grid"))

    elif choice == "3":
        # Parse date
        df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors="coerce")
        df["Year-Quarter"] = df["Fecha finalización"].dt.to_period("Q").astype(str)

        # Count ERRORs per responsable per quarter
        error_trend = (
            df[df["Resultado"] == "ERROR"]
            .groupby(["Year-Quarter", "Responsable Inicio"])
            .size()
            .reset_index(name="ERROR Count")
        )

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=error_trend, x="Year-Quarter", y="ERROR Count", hue="Responsable Inicio", marker="o")
        plt.title("ERROR Trend Over Time by Responsable")
        plt.ylabel("ERROR Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid option. Please enter 1, 2, or 3.")

#Option 10: 
def deflection_boxplot_per_quarter(df):
    df = df.copy()

    # Fixed variable
    var_name = "Defl. [mm]"
    ylabel = "Deflection [mm]"

    # Normalize and filter
    df["Test Result"] = df.iloc[:, 9].astype(str).str.strip().str.upper()
    df = df[df["Test Result"] != "STILL TESTING"]

    # Parse date and create Year-Quarter label
    df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors='coerce')
    df["Year"] = df["Fecha finalización"].dt.year.astype(int)
    df["Quarter"] = df["Fecha finalización"].dt.quarter.astype(int)
    df["Year-Quarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)

    # Prepare data
    df_plot = df[["Year-Quarter", var_name, "Test Result"]].dropna()
    quarter_order = sorted(df_plot["Year-Quarter"].unique(), key=lambda x: pd.Period(x, freq='Q'))

    # Plot
    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(
        data=df_plot,
        x="Year-Quarter",
        y=var_name,
        hue="Test Result",
        order=quarter_order,
        palette={"OK": "green", "ERROR": "blue"}
    )

    # Vertical separators between quarters
    for i in range(len(quarter_order) - 1):
        ax.axvline(x=i + 0.5, linestyle='--', color='gray', alpha=0.5)

    # Annotate per-box sample size
    grouped = df_plot.groupby(["Year-Quarter", "Test Result"])
    for (xval, result), group in grouped:
        x_idx = quarter_order.index(xval)
        hue_offset = -0.2 if result == "OK" else 0.2
        min_val = group[var_name].min()
        label_y = min_val * 0.95
        ax.text(
            x_idx + hue_offset,
            label_y,
            f"n={len(group)}",
            ha='center',
            va='top',
            fontsize=9,
            color='black',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='none', facecolor='white', alpha=0.6)
        )

    # Secondary Y-axis for error rate
    ax2 = ax.twinx()
    grouped_results = df_plot.groupby(["Year-Quarter", "Test Result"]).size().unstack(fill_value=0)
    error_rates = []
    x_positions = []
    for i, quarter in enumerate(quarter_order):
        if quarter == "2021-Q4":
            continue
        if quarter in grouped_results.index:
            total = grouped_results.loc[quarter].sum()
            errors = grouped_results.loc[quarter].get("ERROR", 0)
        else:
            total = 0
            errors = 0
        error_pct = (errors / total) * 100 if total > 0 else 0
        error_rates.append(error_pct)
        x_positions.append(i)

    ax2.plot(
        x_positions,
        error_rates,
        marker='o',
        linestyle='-',
        color='red',
        linewidth=2,
        label='Error Rate (%)'
    )

    for x, y in zip(x_positions, error_rates):
        ax2.text(x, y + 0.3, f"{int(round(y))}%", color='red', fontsize=9, ha='center')

    ax2.set_ylabel("Error Rate (%)", color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 35)

    # Adjust Y-limits (start at 0.5)
    y_max = df_plot[var_name].max()
    y_padding = (y_max - 0.5) * 0.15 if y_max > 0.5 else 1
    ax.set_ylim(0.5, y_max + y_padding)

    # Labels and layout
    ax.set_title("Deflection [mm] Box Plot per Quarter", fontsize=14)
    ax.set_xlabel("Year - Quarter", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(quarter_order)))
    ax.set_xticklabels(quarter_order, rotation=0)
    ax.legend(title="Test Result")
    plt.tight_layout()
    plt.show()


#Option 11
def anillo_analysis(df):
    df = df.copy()

    # Clean and normalize
    df["Resultado"] = df["Resultado"].str.strip().str.upper()
    numeric_cols = ["Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Anillo", "Resultado"] + numeric_cols)

    # Determine if each Anillo had at least one ERROR
    df["Error_bin"] = df["Resultado"].apply(lambda x: 1 if x == "ERROR" else 0)
    anillo_has_error = df.groupby("Anillo")["Error_bin"].max().rename("Error Status")
    anillo_has_error = anillo_has_error.replace({1: "Has ERROR", 0: "No ERROR"})

    # Compute average values per Anillo
    anillo_means = df.groupby("Anillo")[numeric_cols].mean()

    # Combine into a single DataFrame
    summary = pd.concat([anillo_means, anillo_has_error], axis=1).dropna()

    # Prompt user
    print("Choose an analysis option:")
    print("1 - Bar chart: Average variables by Anillo error status")
    print("2 - Logistic Regression: Predict if Anillo will have errors")
    print("3 - Clustering: Group Anillos and compare error status")

    choice = input("Enter option number (1-4): ").strip()

    if choice == "1":
        # Bar chart
        bar_data = summary.groupby("Error Status")[numeric_cols].mean().T
        bar_data.plot(kind='bar', figsize=(10, 6))
        plt.title("Average Values of Variables by Anillo Error Status")
        plt.ylabel("Average Value")
        plt.xlabel("Variable")
        plt.xticks(rotation=0)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    elif choice == "2":
        # Logistic regression
        X = summary[numeric_cols]
        y = (summary["Error Status"] == "Has ERROR").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        print("\nLogistic Regression Performance:\n")
        print(classification_report(y_test, model.predict(X_test)))

        # Coefficients
        coef_df = pd.DataFrame({
            "Variable": numeric_cols,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", key=abs, ascending=False)
        print("\nVariable Importance (Coefficients):")
        print(coef_df)

    elif choice == "3":
        # KMeans clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(summary[numeric_cols])

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        summary["Cluster"] = kmeans.fit_predict(X_scaled)

        crosstab = pd.crosstab(summary["Cluster"], summary["Error Status"])
        print("\nError Status by Cluster:")
        print(crosstab)

        # Optional: visualize cluster centers
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
        centers = scaler.inverse_transform(centers)
        plt.figure(figsize=(10, 6))
        for i, row in enumerate(centers):
            plt.plot(numeric_cols, row, label=f"Cluster {i}")
        plt.title("Cluster Centers (Original Scale)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid option. Please enter 1 to 3.")


#Option 12:
def plot_error_rate_by_variable_bin(df):
    df = df.copy()

    # Column definitions
    col_A = df.columns[0]    # Defl. [mm]
    col_B = df.columns[1]    # Ti [ºC]
    col_C = df.columns[2]    # Tf [ºC]
    col_D = df.columns[3]    # pHi
    col_E = df.columns[4]    # pHf
    col_L = df.columns[11]   # C (%)
    col_M = df.columns[12]   # Cr (%)
    col_N = df.columns[13]   # Mn (%)
    col_O = df.columns[14]   # Mo (%)
    col_P = df.columns[15]   # Sy [MPa]
    col_Q = df.columns[16]   # Uts [MPa]
    col_R = df.columns[17]   # HRC
    col_result = df.columns[9]  # Test Result

    # Variable and bin size mapping for option 1
    variable_map = {
        "1": (col_A, "Defl. [mm]", 0.2),
        "2": (col_B, "Ti [ºC]", 0.5),
        "3": (col_C, "Tf [ºC]", 0.5),
        "4": (col_D, "pHi", 0.05),
        "5": (col_E, "pHf", 0.05),
        "6": (col_L, "C (%)", 0.01),
        "7": (col_M, "Cr (%)", 0.01),
        "8": (col_N, "Mn (%)", 0.01),
        "9": (col_O, "Mo (%)", 0.01),
        "10": (col_P, "Sy [MPa]", 5),
        "11": (col_Q, "Uts [MPa]", 5),
        "12": (col_R, "HRC", 0.5),
    }

    # Clean and prepare data
    df["Resultado"] = df[col_result].astype(str).str.strip().str.upper()
    df = df[df["Resultado"].isin(["OK", "ERROR"])]
    df["Error_bin"] = df["Resultado"].apply(lambda x: 1 if x == "ERROR" else 0)

    # Prompt user
    print("\nWhat would you like to analyze?")
    print("1 - Error rate by bin for a single variable")
    print("2 - Most frequent bin combinations for ERROR results")
    choice = input("Enter option number: ").strip()

    if choice == "1":
        # Show variable options
        print("\nChoose a variable to bin (based on each test value):")
        for key, (_, label, _) in variable_map.items():
            print(f"{key} - {label}")
        var_choice = input("Enter variable number: ").strip()

        if var_choice not in variable_map:
            print("Invalid choice.")
            return

        col_name, label, bin_width = variable_map[var_choice]
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=[col_name])

        # Define bins
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        df["bin"] = pd.cut(df[col_name], bins=bins)

        # Calculate error rate
        error_rate_by_bin = df.groupby("bin")["Error_bin"].mean() * 100
        error_rate_by_bin = error_rate_by_bin.sort_index()

        # Plot
        error_rate_by_bin.plot(kind="bar", figsize=(10, 5))
        plt.title(f"Error Rate by {label} Bins (Per Test)")
        plt.ylabel("Error Rate (%)")
        plt.xlabel(f"{label} Bins")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y", linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    elif choice == "2":
        # Focused analysis on Defl., Sy, and HRC
        bin_settings = {
            "Defl. [mm]": (col_A, 0.2),
            "Sy [MPa]": (col_P, 5),
            "HRC": (col_R, 0.5)
        }

        # Convert and drop missing for relevant columns
        for label, (col, _) in bin_settings.items():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df_clean = df.dropna(subset=[col for col, _ in bin_settings.values()])

        # Create bins
        for label, (col, bin_width) in bin_settings.items():
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            bins = np.arange(min_val, max_val + bin_width, bin_width)
            df_clean[label] = pd.cut(df_clean[col], bins=bins)

        # Group by binned combinations
        bin_labels = list(bin_settings.keys())
        grouped = df_clean.groupby(bin_labels).agg(
            Total=("Resultado", "count"),
            Errors=("Resultado", lambda x: (x == "ERROR").sum())
        ).reset_index()

        grouped["Error Rate (%)"] = (grouped["Errors"] / grouped["Total"]) * 100
        grouped = grouped.sort_values("Errors", ascending=False)

        # Show top combinations with most ERRORs
        print("\nTop combinations with most ERROR test results:\n")
        for _, row in grouped.head(10).iterrows():
            print(f"Amount of ERRORs: {int(row['Errors'])}")
            for label in bin_labels:
                print(f"{label}: {row[label]}")
            print("-" * 40)

        # Show combinations with zero ERRORs
        print("\nCombinations with ZERO ERROR test results:\n")
        zero_errors = grouped[grouped["Errors"] == 0].head(10)
        for _, row in zero_errors.iterrows():
            print(f"Amount of ERRORs: {int(row['Errors'])}")
            for label in bin_labels:
                print(f"{label}: {row[label]}")
            print("-" * 40)



    else:
        print("Invalid option.")




#Option 13
def less_than_100_hs_analysis(df):
    df = df.copy()

    # Normalize data
    df["Horas Progreso"] = pd.to_numeric(df["Horas Progreso"], errors="coerce")
    df["Fecha ensayo"] = pd.to_datetime(df["Fecha ensayo"], errors="coerce")

    chem_cols = ["C (%)", "Cr (%)", "Mn (%)", "Mo (%)", "S (%)", "P (%)"]
    for col in chem_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create SOHIC label
    df["SOHIC"] = df["Horas Progreso"] < 100

    # Drop rows with missing values
    df = df.dropna(subset=chem_cols + ["Horas Progreso", "Fecha ensayo"])

    # Prompt user
    print("Choose an analysis option:")
    print("1 - Boxplots of chemical composition by SOHIC")
    print("2 - Trend over time for chemical elements with Horas Progreso < 100 (SOHIC only)")
    option = input("Enter option (1 or 2): ").strip()

    if option == "1":
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, element in enumerate(chem_cols):
            sns.boxplot(data=df, x="SOHIC", y=element, ax=axes[i])
            axes[i].set_title(f'{element} vs SOHIC')
            axes[i].set_xlabel("")
            axes[i].set_ylabel(element)
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(["Non-SOHIC", "SOHIC"])
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

    elif option == "2":
        df_sohic = df[df["SOHIC"] == True]

        # Melt for Plotly
        melted = df_sohic.melt(
            id_vars="Fecha ensayo",
            value_vars=chem_cols,
            var_name="Elemento",
            value_name="Valor"
        )

        # Build one trace per element (but only make one visible initially)
        fig = px.line(
            melted[melted["Elemento"] == chem_cols[0]].sort_values("Fecha ensayo"),
            x="Fecha ensayo",
            y="Valor",
            title=f"Trend of {chem_cols[0]} over Time (SOHIC only)"
        )

        # Add all other elements as traces (hidden initially)
        for element in chem_cols[1:]:
            temp_df = melted[melted["Elemento"] == element].sort_values("Fecha ensayo")
            fig.add_scatter(
                x=temp_df["Fecha ensayo"],
                y=temp_df["Valor"],
                mode="lines",
                name=element,
                visible=False
            )

        # Create dropdown menu to toggle visibility
        buttons = []
        for i, element in enumerate(chem_cols):
            visibility = [j == i for j in range(len(chem_cols))]
            buttons.append(
                dict(label=element,
                     method="update",
                     args=[{"visible": visibility},
                           {"title": f"Trend of {element} over Time (SOHIC only)",
                            "yaxis": {"title": f"{element}"}}])
            )

        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.15,
                "y": 1.2,
            }],
            yaxis_title=chem_cols[0],
            xaxis_title="Fecha ensayo"
        )

        fig.show()

    else:
        print("Invalid option. Please choose 1 or 2.")


#Option 14
def plot_error_counts_by_quarter(df):
    print("Choose an analysis option:")
    print("1 - Bar chart: Resultado por HTR (Q1-Q2 2025)")
    print("2 - Statistical comparison: OK vs ERROR (Q1-Q2 2025)")
    print("3 - Average time between steps (ERROR vs OK)")
    print("4 - Check errors for Almacenamiento")
    print("5 - Analyze by Lado")
    print("6 - Analyze HTR by horas progreso")

    option = input("Enter option: ").strip()

    if option == "1":
        df = df.copy()

        # Normalize relevant columns
        df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors="coerce")
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
        df["Horas Progreso"] = pd.to_numeric(df["Horas Progreso"], errors="coerce")

        # Only include "OK" and "ERROR"
        df = df[df["Resultado"].isin(["OK", "ERROR"])]

        # Get the HTR column (replace with actual name if known)
        col_htr = df.columns[21]

        # Filter to include only HTRs with at least one "ERROR"
        htrs_with_error = df[df["Resultado"] == "ERROR"][col_htr].unique()
        df = df[df[col_htr].isin(htrs_with_error)]

        # Extract year and quarter
        df["Año"] = df["Fecha finalización"].dt.year
        df["Trimestre"] = df["Fecha finalización"].dt.quarter

        # Aggregate counts
        grouped = df.groupby([col_htr, "Resultado", "Año", "Trimestre"]).size().reset_index(name="Cantidad")

        # Build interactive plot
        years = sorted(grouped["Año"].unique())
        quarters = [1, 2, 3, 4]

        # Create figure with dropdown filters
        fig = px.bar(
            grouped,
            x=col_htr,
            y="Cantidad",
            color="Resultado",
            barmode="group",
            title="Resultado por HTR (filtrable por año y trimestre)",
            labels={col_htr: "HTR"}
        )

        # Create dropdown filters for year and quarter
        buttons_year = [
            dict(
                label=str(year),
                method="update",
                args=[{"visible": [(a == year) and (q in grouped["Trimestre"].unique()) for a, q in zip(grouped["Año"], grouped["Trimestre"])]}]
            ) for year in years
        ]

        buttons_quarter = [
            dict(
                label=f"Q{q}",
                method="update",
                args=[{"visible": [(q == t) for t in grouped["Trimestre"]]}]
            ) for q in quarters
        ]

        # Add dropdown menus
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons_year,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    name="Año"
                ),
                dict(
                    buttons=buttons_quarter,
                    direction="down",
                    showactive=True,
                    x=0.4,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    name="Trimestre"
                )
            ]
        )

        fig.show()



    elif option == "2":
        df = df.copy()

        # Normalize columns
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
        df["Fecha ensayo"] = pd.to_datetime(df["Fecha ensayo"], errors="coerce")

        # Filter for tests in 2025
        df_2025 = df[df["Fecha ensayo"].dt.year == 2025].copy()

        # Variables to analyze
        variables = [
            "Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf",
            "C (%)", "Cr (%)", "Mn (%)", "Mo (%)",
            "Sy [Mpa]", "Uts [MPa]", "HRC", "E ind. [J]", "E prom. [J]"
        ]

        print("\n--- Global Statistical Comparison: OK vs ERROR Test Results (2025) ---\n")
        table = []

        for var in variables:
            if var not in df_2025.columns:
                continue

            df_2025[var] = pd.to_numeric(df_2025[var], errors="coerce")

            group_ok = df_2025[df_2025["Resultado"] == "OK"][var].dropna()
            group_err = df_2025[df_2025["Resultado"] == "ERROR"][var].dropna()

            if len(group_ok) < 3 or len(group_err) < 3:
                continue

            mean_ok, std_ok, n_ok = group_ok.mean(), group_ok.std(), len(group_ok)
            mean_err, std_err, n_err = group_err.mean(), group_err.std(), len(group_err)

            stat, p_val = mannwhitneyu(group_ok, group_err, alternative='two-sided')
            significance = "✅ Significant" if p_val < 0.05 else "❌ Not significant"

            table.append([
                var,
                f"{mean_ok:.2f} ± {std_ok:.2f} (n={n_ok})",
                f"{mean_err:.2f} ± {std_err:.2f} (n={n_err})",
                f"{p_val:.4f}",
                significance
            ])

        headers = ["Variable", "OK Mean ± Std (n)", "ERROR Mean ± Std (n)", "p-value", "Significance"]
        print(tabulate(table, headers=headers, tablefmt="github"))

    elif option == "3":
        df = df.copy()
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()

        # Sub-option: filter by Q1-Q2 2025 or not
        print("\nDo you want to analyze:")
        print("1 - All test results")
        print("2 - Only tests completed in Q1 and Q2 of 2025")
        suboption = input("Enter sub-option (1 or 2): ").strip()

        if suboption not in ["1", "2"]:
            print("Invalid sub-option. Please enter 1 or 2.")
            return

        # Convert relevant date columns to datetime
        date_cols = [
            "Fecha colada", "Fecha creación", "Fecha lista",
            "Fecha entrada labo", "Fecha lanzamiento ensayo",
            "Fecha ensayo", "Fecha TT"
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")

        # Apply filtering if user selects Q1/Q2 2025
        if suboption == "2":
            df = df[
                (df["Fecha ensayo"].dt.year == 2025) &
                (df["Fecha ensayo"].dt.quarter.isin([1, 2]))
            ]
            title_main = "--- Average Time Durations (Q1-Q2 2025) ---"
            title_extra = "--- Additional Time Durations (Q1-Q2 2025) ---"
        else:
            title_main = "--- Average Time Durations (All Test Results) ---"
            title_extra = "--- Additional Time Durations (All Test Results) ---"

        # Compute time differences
        df["colada→creacion"] = (df["Fecha creación"] - df["Fecha colada"]).dt.days
        df["creacion→lista"] = (df["Fecha lista"] - df["Fecha creación"]).dt.days
        df["lista→entrada"] = (df["Fecha entrada labo"] - df["Fecha lista"]).dt.days
        df["entrada→lanzamiento"] = (df["Fecha lanzamiento ensayo"] - df["Fecha entrada labo"]).dt.days
        df["lanzamiento→ensayo"] = (df["Fecha ensayo"] - df["Fecha lanzamiento ensayo"]).dt.days

        # Define steps
        time_steps = [
            ("colada→creacion", "Cast → Cut"),
            ("creacion→lista", "Cut → Lathe"),
            ("lista→entrada", "Lathe → Lab Arrival"),
            ("entrada→lanzamiento", "Lab Arrival → Launch"),
            ("lanzamiento→ensayo", "Launch → Test End")
        ]

        # Collect average durations and compute deltas
        results_by_step = {}
        for result in ["OK", "ERROR"]:
            results_by_step[result] = {}
            for step_col, _ in time_steps:
                subset = df[df["Resultado"] == result][step_col].dropna()
                if len(subset) >= 3:
                    results_by_step[result][step_col] = {
                        "mean": subset.mean(),
                        "n": len(subset)
                    }

        table = []
        for step_col, label in time_steps:
            ok_data = results_by_step.get("OK", {}).get(step_col)
            err_data = results_by_step.get("ERROR", {}).get(step_col)

            if ok_data and err_data:
                delta = err_data["mean"] - ok_data["mean"]
                table.append([
                    label,
                    "OK",
                    f"{ok_data['mean']:.2f} days",
                    "",  # delta blank for OK
                    f"n={ok_data['n']}"
                ])
                table.append([
                    label,
                    "ERROR",
                    f"{err_data['mean']:.2f} days",
                    f"{delta:+.2f} days",  # show + or - delta
                    f"n={err_data['n']}"
                ])

        headers = ["Step", "Resultado", "Avg Duration (days)", "Δ ERROR–OK", "Sample Size"]
        print(f"\n{title_main}\n")
        print(tabulate(table, headers=headers, tablefmt="github"))

        # Additional steps: colada→TT and TT→lanzamiento
        if {"Fecha TT", "Fecha colada"}.issubset(df.columns):
            df["colada→TT"] = (df["Fecha TT"] - df["Fecha colada"]).dt.days
            df["TT→lanzamiento"] = (df["Fecha lanzamiento ensayo"] - df["Fecha TT"]).dt.days

            extra_steps = [
                ("colada→TT", "Cast → Heat Treatment"),
                ("TT→lanzamiento", "Heat Treatment → Launch")
            ]

            results_extra = {}
            for result in ["OK", "ERROR"]:
                results_extra[result] = {}
                for col_name, _ in extra_steps:
                    subset = df[df["Resultado"] == result][col_name].dropna()
                    if len(subset) >= 3:
                        results_extra[result][col_name] = {
                            "mean": subset.mean(),
                            "n": len(subset)
                        }

            table_extra = []
            for col_name, label in extra_steps:
                ok_data = results_extra.get("OK", {}).get(col_name)
                err_data = results_extra.get("ERROR", {}).get(col_name)

                if ok_data and err_data:
                    delta = err_data["mean"] - ok_data["mean"]
                    table_extra.append([
                        label,
                        "OK",
                        f"{ok_data['mean']:.2f} days",
                        "",
                        f"n={ok_data['n']}"
                    ])
                    table_extra.append([
                        label,
                        "ERROR",
                        f"{err_data['mean']:.2f} days",
                        f"{delta:+.2f} days",
                        f"n={err_data['n']}"
                    ])

            print(f"\n{title_extra}\n")
            print(tabulate(table_extra, headers=headers, tablefmt="github"))
        else:
            print("\nColumns 'Fecha colada' and/or 'Fecha TT' not found. Skipping those comparisons.")


    elif option == "4":
        df = df.copy()
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
        col_storage = "Almacenamiento"

        if col_storage not in df.columns:
            print(f"\n❌ Column '{col_storage}' not found in the dataset.")
            return

        # Count ERROR results per Almacenamiento
        error_counts = (
            df[df["Resultado"] == "ERROR"]
            [col_storage]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Almacenamiento", col_storage: "ERROR Count"})
        )

        if error_counts.empty:
            print("\n⚠️ No ERROR results found.")
            return

        print("\n--- ERROR Counts by Almacenamiento ---\n")
        print(tabulate(error_counts, headers="keys", tablefmt="github"))

    elif option == "5":
        df = df.copy()
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
        df["Fecha ensayo"] = pd.to_datetime(df["Fecha ensayo"], errors="coerce")

        col_lado = "Lado"

        if col_lado not in df.columns:
            print(f"\n❌ Column '{col_lado}' not found in the dataset.")
            return

        # Prompt user for date filtering
        print("\nDo you want to analyze:")
        print("1 - All test results (global)")
        print("2 - Only test results from Q1 and Q2 of 2025")
        subopt = input("Enter sub-option (1 or 2): ").strip()

        if subopt == "2":
            df = df[
                (df["Fecha ensayo"].dt.year == 2025) &
                (df["Fecha ensayo"].dt.quarter.isin([1, 2]))
            ]

        # Get all unique "Lado" values
        all_lados = df[[col_lado]].dropna().drop_duplicates()

        # Count ERRORs
        error_counts = (
            df[df["Resultado"] == "ERROR"]
            .groupby(col_lado)
            .size()
            .reset_index(name="ERROR Count")
        )

        # Count OKs
        ok_counts = (
            df[df["Resultado"] == "OK"]
            .groupby(col_lado)
            .size()
            .reset_index(name="OK Count")
        )

        # Merge all
        merged = pd.merge(all_lados, error_counts, on=col_lado, how="left")
        merged = pd.merge(merged, ok_counts, on=col_lado, how="left")
        merged["ERROR Count"] = merged["ERROR Count"].fillna(0).astype(int)
        merged["OK Count"] = merged["OK Count"].fillna(0).astype(int)

        # Optional: sort by ERROR count
        merged = merged.sort_values("ERROR Count", ascending=False)

        print("\n--- Test Result Counts by Lado ---\n")
        print(tabulate(merged, headers=["Lado", "ERROR Count", "OK Count"], tablefmt="github"))

    elif option == "6":
        df = df.copy()

        # Normalize and clean columns
        df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
        df["Horas Progreso"] = pd.to_numeric(df["Horas Progreso"], errors="coerce")

        # Exclude "STILL TESTING"
        df = df[~df["Resultado"].str.contains("STILL TESTING")]

        # Define HTR column (replace with actual name if known)
        col_htr = df.columns[21]  # e.g., "HTR"

        # Identify HTRs with ≥ 2 ERRORs
        htr_error_counts = (
            df[df["Resultado"] == "ERROR"]
            [col_htr]
            .value_counts()
        )
        htrs_with_2_errors = htr_error_counts[htr_error_counts >= 2].index

        # Filter to only these HTRs
        df_filtered = df[df[col_htr].isin(htrs_with_2_errors)]

        # Create bins for Horas Progreso
        df_filtered["Bin"] = pd.cut(
            df_filtered["Horas Progreso"],
            bins=[-float("inf"), 100, 720],
            labels=["<100 hs", "100-720 hs"],
            right=False
        )

        # Count per HTR and Bin
        summary = (
            df_filtered
            .groupby([col_htr, "Bin"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={
                "<100 hs": "Count (<100 hs)",
                "100-720 hs": "Count (100-720 hs)"
            })
        )

        print("\n--- Test Counts by Horas Progreso for HTRs with ≥ 2 ERROR Results (Excluding 'STILL TESTING') ---\n")
        print(tabulate(summary, headers="keys", tablefmt="github"))

    else:
        print("Invalid option. Please enter a valid option")

#Option 15
def analyze_by_lado(df):
    df = df.copy()
    df["Lado"] = df["Lado"].astype(str).str.strip().str.upper()

    # Prompt user for variable to analyze
    print("\nWhich property would you like to visualize by Lado?")
    print("1 - Sy [MPa]")
    print("2 - HRC")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        column = "Sy [Mpa]"
    elif choice == "2":
        column = "HRC"
    else:
        print("❌ Invalid selection.")
        return

    # Prepare the data
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["Lado"].isin(["ESTE", "OESTE"])]
    df = df[df[column].notna()]

    if df.empty:
        print(f"\n⚠️ No valid '{column}' data found for Lado ESTE or OESTE.")
        return

    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue="Lado", kde=True, bins=20, element="step", stat="density")
    plt.title(f"Distribution of {column} for Lado: ESTE vs OESTE")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



#Option 16
def analyze_boxplots(df):
    df = df.copy()

    # Clean columns
    df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
    df = df[df["Resultado"].isin(["OK", "ERROR"])]

    # Convert Fecha finalización
    df["Fecha finalización"] = pd.to_datetime(df["Fecha finalización"], errors="coerce")
    df = df.dropna(subset=["Fecha finalización"])

    # Create Quarter-Year
    df["Quarter-Year"] = df["Fecha finalización"].dt.to_period("Q").astype(str)

    # Create a sorting key for quarters
    df["QuarterStart"] = df["Fecha finalización"].dt.to_period("Q").dt.start_time
    quarter_order = df.drop_duplicates("Quarter-Year").sort_values("QuarterStart")["Quarter-Year"].tolist()

    # Variables to include
    variables = [
        "Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf",
        "C (%)", "Cr (%)", "Mn (%)", "Mo (%)", "Sy [Mpa]",
        "Uts [MPa]", "HRC", "E ind. [J]", "E prom. [J]"
    ]

    # Ensure numeric
    for col in variables:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create traces for each variable (OK and ERROR boxplots)
    fig = go.Figure()

    for i, var in enumerate(variables):
        visible = True if i == 0 else False
        for result in ["OK", "ERROR"]:
            filtered = df[df["Resultado"] == result]
            fig.add_trace(go.Box(
                y=filtered[var],
                x=filtered["Quarter-Year"],
                name=result,
                boxpoints=False,
                visible=visible,
                legendgroup=result,
                marker_color="blue" if result == "OK" else "red"
            ))

    # Calculate error rate per quarter
    total_counts = df.groupby("Quarter-Year")["Resultado"].value_counts().unstack().fillna(0)
    total_counts = total_counts.reindex(quarter_order)  # Ensure proper order
    total_counts["Error Rate (%)"] = (total_counts.get("ERROR", 0) / 
                                      (total_counts.get("OK", 0) + total_counts.get("ERROR", 0))) * 100

    # Exclude 2021Q4 from error rate line
    error_line_x = [q for q in quarter_order if q != "2021Q4"]
    error_line_y = total_counts.loc[total_counts.index != "2021Q4", "Error Rate (%)"]

    # Add error rate line trace
    fig.add_trace(go.Scatter(
        x=error_line_x,
        y=error_line_y,
        mode="lines+markers",
        name="Error Rate (%)",
        yaxis="y2",
        line=dict(color="red", width=2, dash="solid"),
        marker=dict(symbol="circle", size=8)
    ))

    # Create dropdown menu for variable selection
    dropdown_buttons = []
    for i, var in enumerate(variables):
        visibility = [False] * (len(variables) * 2)
        visibility[i * 2] = True   # OK
        visibility[i * 2 + 1] = True # ERROR
        visibility.append(True)  # Keep error rate line always visible
        dropdown_buttons.append(dict(
            label=var,
            method="update",
            args=[
                {"visible": visibility},
                {"yaxis": {"title": var},
                 "title": f"{var} by Quarter-Year and Resultado"}
            ]
        ))

    # Final layout
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            x=1.1,
            y=1.15
        )],
        title=f"{variables[0]} by Quarter-Year and Resultado",
        xaxis=dict(
            title="Quarter-Year",
            categoryorder="array",
            categoryarray=quarter_order
        ),
        yaxis=dict(
            title=variables[0]
        ),
        yaxis2=dict(
            title="Error Rate (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickformat=".0f",
            range=[0, 30]  # Limit to 30%
        ),
        boxmode="group",
        height=600
    )

    fig.show()


#option 17:
def analyze_hrc_16849_differences(df):
    df = df.copy()

    # Normalize columns
    df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
    df = df[df["Resultado"].isin(["OK", "ERROR"])]

    # Filter only for HRC == 16849
    if "HTR" not in df.columns:
        print("Column 'HRC' not found.")
        return

    df = df[pd.to_numeric(df["HTR"], errors="coerce") == 16849]

    # Variables to compare
    variables = [
        "Defl. [mm]", "Ti [ºC]", "Tf [ºC]", "pHi", "pHf",
        "C (%)", "Cr (%)", "Mn (%)", "Mo (%)", "Sy [Mpa]",
        "Uts [MPa]", "E ind. [J]", "E prom. [J]"
    ]

    for var in variables:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    # Group by Resultado
    summary = []
    for var in variables:
        ok_values = df[df["Resultado"] == "OK"][var].dropna()
        error_values = df[df["Resultado"] == "ERROR"][var].dropna()
        
        if len(ok_values) == 0 or len(error_values) == 0:
            continue
        
        mean_ok = ok_values.mean()
        mean_error = error_values.mean()
        std_ok = ok_values.std()
        std_error = error_values.std()
        diff = mean_error - mean_ok
        
        summary.append([
            var,
            f"{mean_ok:.2f}",
            f"{mean_error:.2f}",
            f"{std_ok:.2f}",
            f"{std_error:.2f}",
            f"{diff:+.2f}"
        ])

    headers = ["Variable", "Mean OK", "Mean ERROR", "Std OK", "Std ERROR", "Δ ERROR - OK"]
    print(tabulate(summary, headers=headers, tablefmt="fancy_grid"))


#Option 18
def analyze_coladas(df):
    df = df.copy()

    # Ask user for date column to use
    print("Choose the date field to analyze by:")
    print("1 - Fecha colada")
    print("2 - Fecha lanzamiento ensayo")
    date_choice = input("Enter 1 or 2: ").strip()

    if date_choice == "2":
        date_col = "Fecha lanzamiento ensayo"
    else:
        date_col = "Fecha colada"

    # Clean and standardize
    df["Resultado"] = df["Resultado"].astype(str).str.strip().str.upper()
    df = df[df["Resultado"].isin(["OK", "ERROR"])]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    ### FIRST TABLE: Top Dates by ERROR Rate
    fecha_summary = (
        df.groupby(date_col)["Resultado"]
        .value_counts()
        .unstack(fill_value=0)
        .assign(Total=lambda x: x["OK"] + x["ERROR"])
        .assign(ErrorRate=lambda x: x["ERROR"] / x["Total"])
    )

    top_dates = (
        fecha_summary[fecha_summary["Total"] >= 5]
        .sort_values(by="ErrorRate", ascending=False)
        .head(10)
        .index
    )

    error_rows = []

    for fecha in top_dates:
        subset = df[df[date_col] == fecha]
        colada_group = (
            subset.groupby("Colada")["Resultado"]
            .value_counts()
            .unstack(fill_value=0)
            .assign(Total=lambda x: x["OK"] + x["ERROR"])
            .assign(ErrorRate=lambda x: x["ERROR"] / x["Total"])
            .reset_index()
        )

        formatted_date = f"{fecha.date()} ({fecha.strftime('%A')})"
        first_row = True

        for _, row in colada_group.iterrows():
            error_rows.append({
                date_col: formatted_date if first_row else "",
                "Colada": row["Colada"],
                "OK": int(row["OK"]),
                "ERROR": int(row["ERROR"]),
                "Total": int(row["Total"]),
                "ErrorRate": round(row["ErrorRate"], 2)
            })
            first_row = False

    ### SECOND TABLE: Coladas with NO ERROR
    grouped = (
        df.groupby([date_col, "Colada"])["Resultado"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    grouped["OK"] = grouped.get("OK", 0)
    grouped["ERROR"] = grouped.get("ERROR", 0)
    grouped["Total"] = grouped["OK"] + grouped["ERROR"]
    grouped["ErrorRate"] = grouped["ERROR"] / grouped["Total"]

    no_error = grouped[(grouped["ERROR"] == 0) & (grouped["Total"] >= 5)]
    no_error = no_error.sort_values(by="Total", ascending=False).head(30)

    no_error_rows = []
    current_date = None

    for _, row in no_error.iterrows():
        fecha = row[date_col]
        formatted_date = f"{fecha.date()} ({fecha.strftime('%A')})" if fecha != current_date else ""
        current_date = fecha

        no_error_rows.append({
            date_col: formatted_date,
            "Colada": row["Colada"],
            "OK": int(row["OK"]),
            "ERROR": int(row["ERROR"]),
            "Total": int(row["Total"]),
            "ErrorRate": round(row["ErrorRate"], 2)
        })

    ### PRINT BOTH TABLES
    print(f"\n🔴 Coladas in Top Dates by ERROR rate (based on {date_col}):")
    print(tabulate(error_rows, headers="keys", tablefmt="fancy_grid", showindex=False))

    print(f"\n🟢 Top Coladas with NO ERRORs (OK-only results, Total ≥ 5) (based on {date_col}):")
    print(tabulate(no_error_rows, headers="keys", tablefmt="fancy_grid", showindex=False))

    ### DAY-OF-WEEK SUMMARY FOR OK AND ERROR
    weekday_summary = (
        df.groupby(df[date_col].dt.day_name())["Resultado"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
    )

    weekday_summary["Total"] = weekday_summary["OK"] + weekday_summary["ERROR"]

    most_error_day = weekday_summary["ERROR"].idxmax()
    most_error_val = weekday_summary.loc[most_error_day, "ERROR"]
    most_error_total = weekday_summary.loc[most_error_day, "Total"]

    least_error_day = weekday_summary["ERROR"].idxmin()
    least_error_val = weekday_summary.loc[least_error_day, "ERROR"]
    least_error_total = weekday_summary.loc[least_error_day, "Total"]

    most_ok_day = weekday_summary["OK"].idxmax()
    most_ok_val = weekday_summary.loc[most_ok_day, "OK"]
    most_ok_total = weekday_summary.loc[most_ok_day, "Total"]

    least_ok_day = weekday_summary["OK"].idxmin()
    least_ok_val = weekday_summary.loc[least_ok_day, "OK"]
    least_ok_total = weekday_summary.loc[least_ok_day, "Total"]

    print(f"\n📊 Summary by Day of the Week (based on {date_col}):")
    print(tabulate(weekday_summary.reset_index(), headers="keys", tablefmt="grid", showindex=False))

    print(f"\n🔴 Day with MOST ERROR results: {most_error_day} ({most_error_val} errors out of {most_error_total} samples)")
    print(f"🔴 Day with LEAST ERROR results: {least_error_day} ({least_error_val} errors out of {least_error_total} samples)")
    print(f"🟢 Day with MOST OK results: {most_ok_day} ({most_ok_val} OKs out of {most_ok_total} samples)")
    print(f"🟢 Day with LEAST OK results: {least_ok_day} ({least_ok_val} OKs out of {least_ok_total} samples)")

    ### BAR PLOT: OK and ERROR by day of week
    weekday_plot = weekday_summary[["OK", "ERROR"]].fillna(0)

    ax = weekday_plot.plot(
        kind="bar",
        figsize=(10, 6),
        edgecolor='black'
    )

    plt.title(f"OK and ERROR Test Results by Day of the Week (based on {date_col})")
    plt.xlabel("Day of the Week")
    plt.ylabel("Number of Results")
    plt.xticks(rotation=45)
    plt.legend(title="Test Result")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.show()






    

#Devolucion
def analyze_by_test_result():
    excel_path = r"C:\Users\60098360\Desktop\Excel files\Base de datos TN110SS - Reducido.xlsm"
    if not os.path.isfile(excel_path):
        print(f"File not found: {excel_path}")
        return

    try:
        df = pd.read_excel(excel_path, header=1)

        # Get column J by index (9th column)
        col_j = df.iloc[:, 9]

        # Normalize strings
        col_j = col_j.astype(str).str.strip().str.upper()

        # Count occurrences
        result_counts = col_j.value_counts()

        # Print result
        print("Test result counts in column J:")
        print(result_counts)

        print("\nSelect the graph to display:")
        print("1 - Correlation Heatmaps")
        print("2 - Box plot of yield strength per quarter")
        print("3 - Data Analysis (Group Combinations vs Test Result)")
        print("4 - Time-graphs (Variation of values through time)")
        print("5 - Average of each component per year and quarter")
        print("6 - PCA graph")
        print("7 - Statistical analysis (OK vs ERROR)")
        print("8 - Decision Tree")
        print("9 - Police")
        print("10 - Box plot for deflection per quarter")
        print("11 - ""Anillo"" analysis")
        print("12 - Bin values")
        print("13 - SOHIC analysis")
        print("14 - Analyze material")
        print("15 - Analyze by Lado")
        print("16 - Analyze Boxplots")
        print("17 - Analyze HTR 16849")
        print("18 - Analyze Fechas")

        choice = input("Please enter a number correpsonding to the output needed: ").strip()
        print()

        if choice == "1":
            show_correlation_heatmaps(df)
        elif choice == "2":
            show_yield_strength_by_quarter(df)
        elif choice == "3":
            analyze_data_groups(df)
        elif choice == "4":
            show_time_graph(df)
        elif choice == "5":
            plot_avg_by_quarter(df)
        elif choice == "6":
            print("\nPCA Analysis Options:")
            print("1 - PCA Scatter Plot (PC1 vs PC2)")
            print("2 - PCA Loadings Plot (Variable Contributions)")

            pca_choice = input("Select 1 or 2: ").strip()
            print()

            if pca_choice == "1":
                show_pca_scatter(df)
            elif pca_choice == "2":
                plot_pca_loadings(df)
            else:
                print("Invalid PCA option.")
        elif choice == "7":
            compare_ok_vs_error(df)
        elif choice == "8":
            build_decision_tree(df)
        elif choice == "9":
            analyze_responsable_influence(df)
        elif choice == "10":
            deflection_boxplot_per_quarter(df)
        elif choice == "11":
            anillo_analysis(df)
        elif choice == "12":
            plot_error_rate_by_variable_bin(df)
        elif choice == "13":
            less_than_100_hs_analysis(df)
        elif choice == "14":
            plot_error_counts_by_quarter(df)
        elif choice == "15":
            analyze_by_lado(df)
        elif choice == "16":
            analyze_boxplots(df)
        elif choice == "17":
            analyze_hrc_16849_differences(df)
        elif choice == "18":
            analyze_coladas(df)
        else:
            print("Invalid choice. Please enter a number correpsonding to the output needed: ")

    except Exception as e:
        print(f"Error: {e}")

# Run it
analyze_by_test_result()

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate

# Suppress openpyxl UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# File path
excel_path = input("Enter the full path to the Excel file: ").strip().strip('"').strip("'")



# Sheet options
sheet_options = {
    "1": "Tracción",
    "2": "Dureza",
    "3": "Dureza_anillo_de_temple",
    "4": "Charpy",
    "5": "Método_A"
}

def analyze_traccion(df):
    try:
        # Column references
        col_colada = "Colada"
        col_fuencia = "Fuencia EUL Obt (MPa)"
        col_rotura = "Rotura Obt. (MPa)"

        # Prompt user for coladas
        user_input = input("Enter one or more Colada numbers separated by commas (e.g., 5054,4991): ")
        selected_coladas = [c.strip() for c in user_input.split(",") if c.strip()]

        if not selected_coladas:
            print("⚠️ No coladas entered. Skipping combined analysis.")
        else:
            pattern = "|".join(selected_coladas)
            df_filtered = df[df[col_colada].astype(str).str.contains(pattern, na=False)]

            if not df_filtered.empty:
                fuencia_avg = df_filtered[col_fuencia].mean()
                rotura_avg = df_filtered[col_rotura].mean()
                fuencia_max = df_filtered[col_fuencia].max()
                rotura_max = df_filtered[col_rotura].max()

                print(f"\n--- Combined Statistics for Coladas: {', '.join(selected_coladas)} ---")
                headers1 = ["Metric", "Fluencia EUL Obt (MPa)", "Rotura Obt. (MPa)"]
                data1 = [
                    ["Average", f"{fuencia_avg:.2f}", f"{rotura_avg:.2f}"],
                    ["Maximum", f"{fuencia_max:.2f}", f"{rotura_max:.2f}"]
                ]
                print(tabulate(data1, headers=headers1, tablefmt="grid"))
            else:
                print(f"No matching values found for Coladas: {', '.join(selected_coladas)}")

        # Second analysis: stats per unique colada
        grouped = df.groupby(col_colada, dropna=True)
        data2 = []

        coladas = []
        fuencia_averages = []
        rotura_averages = []

        for colada, group in grouped:
            avg_fuencia = group[col_fuencia].mean()
            max_fuencia = group[col_fuencia].max()
            avg_rotura = group[col_rotura].mean()
            max_rotura = group[col_rotura].max()
            data2.append([
                str(colada),
                f"{avg_fuencia:.2f}",
                f"{max_fuencia:.2f}",
                f"{avg_rotura:.2f}",
                f"{max_rotura:.2f}"
            ])
            coladas.append(str(colada))
            fuencia_averages.append(avg_fuencia)
            rotura_averages.append(avg_rotura)

        # Sort by colada numerically if possible
        try:
            sort_index = sorted(range(len(coladas)), key=lambda i: float(coladas[i]))
            coladas = [coladas[i] for i in sort_index]
            fuencia_averages = [fuencia_averages[i] for i in sort_index]
            rotura_averages = [rotura_averages[i] for i in sort_index]
            data2.sort(key=lambda x: float(x[0]))
        except ValueError:
            pass  # Leave unsorted if coladas are not purely numeric

        headers2 = [
            "Colada",
            "Avg Fluencia (MPa)",
            "Max Fluencia (MPa)",
            "Avg Rotura (MPa)",
            "Max Rotura (MPa)"
        ]

        print("\n--- Statistics by Individual Colada ---")
        print(tabulate(data2, headers=headers2, tablefmt="grid"))

        # Plotting
        x = range(len(coladas))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], fuencia_averages, width=width, label="Avg Fluencia (MPa)")
        plt.bar([i + width/2 for i in x], rotura_averages, width=width, label="Avg Rotura (MPa)")

        plt.xticks(ticks=x, labels=coladas, rotation=0, ha='right')
        plt.xlabel("Colada")
        plt.ylabel("Average Value (MPa)")
        plt.title("Average Fluencia and Rotura by Colada")
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    except KeyError as e:
        print(f"Missing expected column: {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")



def analyze_dureza(df):
    try:
        # Column references
        col_colada = "Colada"
        col_dureza = "Dureza Prom. Max. Obt 1"

        # Prompt user for coladas
        user_input = input("Enter one or more Colada numbers separated by commas (e.g., 5054,4991): ")
        selected_coladas = [c.strip() for c in user_input.split(",") if c.strip()]

        if not selected_coladas:
            print("No coladas entered. Skipping combined analysis.")
        else:
            pattern = "|".join(selected_coladas)
            df_filtered = df[df[col_colada].astype(str).str.contains(pattern, na=False)]

            if not df_filtered.empty:
                dureza_avg = df_filtered[col_dureza].mean()
                dureza_max = df_filtered[col_dureza].max()

                print(f"\n--- Combined Statistics for Coladas: {', '.join(selected_coladas)} ---")
                headers1 = ["Metric", "Dureza HRC"]
                data1 = [
                    ["Average", f"{dureza_avg:.2f}"],
                    ["Maximum", f"{dureza_max:.2f}"]
                ]
                print(tabulate(data1, headers=headers1, tablefmt="grid"))
            else:
                print(f"No matching values found for Coladas: {', '.join(selected_coladas)}")

        # Second analysis: stats per unique colada
        grouped = df.groupby(col_colada, dropna=True)
        data2 = []

        coladas = []
        dureza_averages = []

        for colada, group in grouped:
            avg_dureza = group[col_dureza].mean()
            max_dureza = group[col_dureza].max()
            data2.append([
                str(colada),
                f"{avg_dureza:.2f}",
                f"{max_dureza:.2f}",
            ])
            coladas.append(str(colada))
            dureza_averages.append(avg_dureza)

        # Sort by colada numerically if possible
        try:
            sort_index = sorted(range(len(coladas)), key=lambda i: float(coladas[i]))
            coladas = [coladas[i] for i in sort_index]
            dureza_averages = [dureza_averages[i] for i in sort_index]
            data2.sort(key=lambda x: float(x[0]))
        except ValueError:
            pass  # Leave unsorted if coladas are not purely numeric

        headers2 = [
            "Colada",
            "Avg Dureza (HRC)",
            "Max Dureza (HRC)",
        ]

        print("\n--- Statistics by Individual Colada ---")
        print(tabulate(data2, headers=headers2, tablefmt="grid"))

        # Plotting
        x = range(len(coladas))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], dureza_averages, width=width, label="Avg Dureza (HRC)")

        plt.xticks(ticks=x, labels=coladas, rotation=0, ha='right')
        plt.xlabel("Colada")
        plt.ylabel("Average Value (HRC)")
        plt.title("Average Dureza by Colada")
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    except KeyError as e:
        print(f"Missing expected column: {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")



def analyze_dureza_anillo(df):
    try:
        # Column references
        col_colada = "Colada"
        col_d1 = "Dur Ind. 1"
        col_d2 = "Dur Ind. 2"
        col_d3 = "Dur Ind. 3"
        col_d4 = "Dur Ind. 4"

        # Prompt user for coladas
        user_input = input("Enter one or more Colada numbers separated by commas (e.g., 5054,4991): ")
        selected_coladas = [c.strip() for c in user_input.split(",") if c.strip()]

        if not selected_coladas:
            print("⚠️ No coladas entered. Skipping combined analysis.")
        else:
            pattern = "|".join(selected_coladas)
            df_filtered = df[df[col_colada].astype(str).str.contains(pattern, na=False)]

            if not df_filtered.empty:
                d1_avg = df_filtered[col_d1].mean()
                d2_avg = df_filtered[col_d2].mean()
                d3_avg = df_filtered[col_d3].mean()
                d4_avg = df_filtered[col_d4].mean()

                d1_max = df_filtered[col_d1].max()
                d2_max = df_filtered[col_d2].max()
                d3_max = df_filtered[col_d3].max()
                d4_max = df_filtered[col_d4].max()

                print(f"\n--- Combined Statistics for Coladas: {', '.join(selected_coladas)} ---")
                headers1 = ["Metric", col_d1, col_d2, col_d3, col_d4]
                data1 = [
                    ["Average", f"{d1_avg:.2f}", f"{d2_avg:.2f}", f"{d3_avg:.2f}", f"{d4_avg:.2f}"],
                    ["Maximum", f"{d1_max:.2f}", f"{d2_max:.2f}", f"{d3_max:.2f}", f"{d4_max:.2f}"]
                ]
                print(tabulate(data1, headers=headers1, tablefmt="grid"))
            else:
                print(f"No matching values found for Coladas: {', '.join(selected_coladas)}")

        # Second analysis: stats per unique colada
        grouped = df.groupby(col_colada, dropna=True)
        data2 = []

        coladas = []
        d1_averages = []
        d2_averages = []
        d3_averages = []
        d4_averages = []

        for colada, group in grouped:
            avg_d1 = group[col_d1].mean()
            max_d1 = group[col_d1].max()
            avg_d2 = group[col_d2].mean()
            max_d2 = group[col_d2].max()
            avg_d3 = group[col_d3].mean()
            max_d3 = group[col_d3].max()
            avg_d4 = group[col_d4].mean()
            max_d4 = group[col_d4].max()
            data2.append([
                str(colada),
                f"{avg_d1:.2f}", f"{max_d1:.2f}",
                f"{avg_d2:.2f}", f"{max_d2:.2f}",
                f"{avg_d3:.2f}", f"{max_d3:.2f}",
                f"{avg_d4:.2f}", f"{max_d4:.2f}"
            ])
            coladas.append(str(colada))
            d1_averages.append(avg_d1)
            d2_averages.append(avg_d2)
            d3_averages.append(avg_d3)
            d4_averages.append(avg_d4)

        # Sort by colada numerically if possible
        try:
            sort_index = sorted(range(len(coladas)), key=lambda i: float(coladas[i]))
            coladas = [coladas[i] for i in sort_index]
            d1_averages = [d1_averages[i] for i in sort_index]
            d2_averages = [d2_averages[i] for i in sort_index]
            d3_averages = [d3_averages[i] for i in sort_index]
            d4_averages = [d4_averages[i] for i in sort_index]
            data2.sort(key=lambda x: float(x[0]))
        except ValueError:
            pass  # Leave unsorted if coladas are not purely numeric

        headers2 = [
            "Colada",
            f"Avg {col_d1}", f"Max {col_d1}",
            f"Avg {col_d2}", f"Max {col_d2}",
            f"Avg {col_d3}", f"Max {col_d3}",
            f"Avg {col_d4}", f"Max {col_d4}"
        ]

        print("\n--- Statistics by Individual Colada ---")
        print(tabulate(data2, headers=headers2, tablefmt="grid"))

        # Plotting
        x = range(len(coladas))
        width = 0.18

        plt.figure(figsize=(14, 6))
        plt.bar([i - 1.5 * width for i in x], d1_averages, width=width, label=col_d1)
        plt.bar([i - 0.5 * width for i in x], d2_averages, width=width, label=col_d2)
        plt.bar([i + 0.5 * width for i in x], d3_averages, width=width, label=col_d3)
        plt.bar([i + 1.5 * width for i in x], d4_averages, width=width, label=col_d4)

        plt.xticks(ticks=x, labels=coladas, rotation=0, ha='right')
        plt.xlabel("Colada")
        plt.ylabel("Average Hardness")
        plt.title("Average Dureza Individual by Colada")
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    except KeyError as e:
        print(f"Missing expected column: {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")



def analyze_charpy_energy(df):
    try:
        # Column references
        col_colada = "Colada"
        col_energy = "Energía Prom. Mín Obt. (J)"

        # Prompt user for coladas
        user_input = input("Enter one or more Colada numbers separated by commas (e.g., 5054,4991): ")
        selected_coladas = [c.strip() for c in user_input.split(",") if c.strip()]

        if not selected_coladas:
            print("⚠️ No coladas entered. Skipping combined analysis.")
        else:
            pattern = "|".join(selected_coladas)
            df_filtered = df[df[col_colada].astype(str).str.contains(pattern, na=False)]

            if not df_filtered.empty:
                energy_avg = df_filtered[col_energy].mean()
                energy_max = df_filtered[col_energy].max()

                print(f"\n--- Combined Statistics for Coladas: {', '.join(selected_coladas)} ---")
                headers1 = ["Metric", col_energy]
                data1 = [
                    ["Average", f"{energy_avg:.2f}"],
                    ["Maximum", f"{energy_max:.2f}"]
                ]
                print(tabulate(data1, headers=headers1, tablefmt="grid"))
            else:
                print(f"No matching values found for Coladas: {', '.join(selected_coladas)}")

        # Second analysis: stats per individual colada
        grouped = df.groupby(col_colada, dropna=True)
        data2 = []

        coladas = []
        energy_averages = []

        for colada, group in grouped:
            avg_energy = group[col_energy].mean()
            max_energy = group[col_energy].max()
            data2.append([
                str(colada),
                f"{avg_energy:.2f}",
                f"{max_energy:.2f}"
            ])
            coladas.append(str(colada))
            energy_averages.append(avg_energy)

        # Sort by colada numerically if possible
        try:
            sort_index = sorted(range(len(coladas)), key=lambda i: float(coladas[i]))
            coladas = [coladas[i] for i in sort_index]
            energy_averages = [energy_averages[i] for i in sort_index]
            data2.sort(key=lambda x: float(x[0]))
        except ValueError:
            pass  # Leave unsorted if coladas are not purely numeric

        headers2 = ["Colada", f"Avg {col_energy}", f"Max {col_energy}"]

        print("\n--- Statistics by Individual Colada ---")
        print(tabulate(data2, headers=headers2, tablefmt="grid"))

        # Plotting
        x = range(len(coladas))
        plt.figure(figsize=(12, 6))
        plt.bar(x, energy_averages, width=0.4, label=col_energy)

        plt.xticks(ticks=x, labels=coladas, rotation=0, ha='right')
        plt.xlabel("Colada")
        plt.ylabel("Average Energy (J)")
        plt.title("Average Charpy Energy by Colada")
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    except KeyError as e:
        print(f"Missing expected column: {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")



def main():
    print("Select a sheet to read:")
    for key, name in sheet_options.items():
        print(f"{key}. {name}")
    
    choice = input("Enter the number corresponding to your choice: ").strip()
    
    if choice in sheet_options:
        selected_sheet = sheet_options[choice]
        print(f"\nLoading sheet: {selected_sheet}")
        
        try:
            df = pd.read_excel(excel_path, sheet_name=selected_sheet)

            if selected_sheet == "Tracción":
                analyze_traccion(df)
            elif selected_sheet == "Dureza":
                analyze_dureza(df)
            elif selected_sheet == "Dureza_anillo_de_temple":
                analyze_dureza_anillo(df)
            elif selected_sheet == "Charpy":
                analyze_charpy_energy(df)
            else:
                print(f"Placeholder for processing the '{selected_sheet}' sheet.")
                print(df.head())
        
        except Exception as e:
            print(f"An error occurred while reading the sheet: {e}")
    else:
        print("Invalid selection. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()

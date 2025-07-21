import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Data ---
hardness_data = [
    19.73, 20, 18.27, 18.43, 19.93, 18.67, 18.2, 19.1,
    19.33, 18.77, 19.2, 19.33, 18.2, 19.1, 19.33, 18.77,
    19.2, 19.33, 18.17, 19.87, 19.03, 19.8, 19.3, 19.6
]

yield_strength_data = [
    614.582, 634.379, 603, 610, 622, 609, 597, 615, 607,
    610, 589, 611, 597, 615, 607, 610, 589, 611, 604,
    623, 614, 603, 605, 591
]

uts_data = [
    712, 709, 715, 698, 730, 705, 699, 690, 720, 710,
    700, 685, 735, 695, 705, 715, 700, 695, 705, 720,
    690, 725, 710, 700
]

# --- Prompt user ---
option = input("Choose dataset to plot:\n1 = Hardness\n2 = EUL Yield Strength\n3 = UTS\nEnter option (1, 2 or 3): ")

# --- Assign based on selection ---
if option == "1":
    data = hardness_data
    title = "Histogram of Hardness [HRC]"
    x_label = "Hardness (HRC)"
    bins = np.arange(18, 21.25, 0.25)
    vertical_lines = [22]
    x_min, x_max = 17, 22.5  
elif option == "2":
    data = yield_strength_data
    title = "Histogram of EUL Yield Strength [MPa]"
    x_label = "Yield Strength (MPa)"
    bins = np.arange(580, 660 + 5, 5)
    vertical_lines = [552, 655]
    x_min, x_max = 540, 670
elif option == "3":
    data = uts_data
    title = "Histogram of UTS [MPa]"
    x_label = "Ultimate Tensile Strength (MPa)"
    bins = np.arange(680, 740 + 5, 5)
    vertical_lines = [655]
    x_min, x_max = 640, 780
else:
    print("Invalid option.")
    exit()

# --- Stats ---
mean = np.mean(data)
std = np.std(data)

# --- Plot ---
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=bins, stat="density", color="#007acc", edgecolor="black", alpha=0.8)

# KDE line
sns.kdeplot(data, color="red", linewidth=2, clip=(x_min, x_max))

# Vertical lines
for x in vertical_lines:
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1.5)

# Stats box
textstr = f"Mean = {mean:.2f}\nStdDev = {std:.2f}\nn = {len(data)}"
props = dict(boxstyle='round', facecolor='white', edgecolor='black')
plt.text(0.98, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=props)

# Labels
plt.title(title, fontsize=14)
plt.xlabel(x_label)
plt.ylabel("Density")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(x_min, x_max)
plt.tight_layout()
plt.show()

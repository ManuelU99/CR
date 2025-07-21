import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data from the Excel file (replace with your actual path)
file_path = r"C:\Users\60098360\Desktop\Excel files\Test.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Limit to rows 3 to 1090 (index 2 to 1090) and filter for 'OK' or 'ERROR' in column J (index 9)
subset_df = df.iloc[2:1090]
filtered_df = subset_df[subset_df.iloc[:, 9].isin(['OK', 'ERROR'])]

# Extract variables (columns A to E) and test results (column J)
variables = filtered_df.iloc[:, 0:5]
test_results = filtered_df.iloc[:, 9]

# Standardize the variables
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(variables_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Test Result'] = test_results.values

# Plot the PCA results
plt.figure(figsize=(10, 7))
colors = {'OK': 'blue', 'ERROR': 'red'}
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Test Result'].map(colors), alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Test Results')
plt.grid(True)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='OK', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='ERROR', markerfacecolor='red', markersize=10)
], title='Test Result')
plt.show()

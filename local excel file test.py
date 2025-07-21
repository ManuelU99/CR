import pandas as pd
import os

local_csv_path = r"C:\Users\60098360\Desktop\Python codes\Graph Quality Control Check.csv"

entry = pd.DataFrame([{
    "Familia": "TestFam",
    "Tipo_Acero_Limpio": "TestAcero",
    "Ciclo": "TestCiclo",
    "Soaking": "TestSoak",
    "GroupNumber": "1",
    "TestType": "Traccion",
    "IsCorrect": "No",
    "Reason": "Test reason entry"
}])

if os.path.exists(local_csv_path):
    print("File exists — reading existing data.")
    existing = pd.read_csv(local_csv_path)
    updated = pd.concat([existing, entry], ignore_index=True)
    updated.to_csv(local_csv_path, index=False)
    print("✅ Data appended and saved to CSV.")
else:
    entry.to_csv(local_csv_path, index=False)
    print("✅ New CSV created and saved.")

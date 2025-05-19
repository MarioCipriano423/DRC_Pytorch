import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Cargar etiquetas originales
print(f"Leyendo etiquetas...")
LABELS_PATH = 'data/labels.csv'  # cambia esta línea
df = pd.read_csv(LABELS_PATH)

# 2. Primer split: train (80%) y resto (20%)
print(f"Train split...")
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# 3. Segundo split: test (10%) y predict (10%) a partir del resto
print(f"Test-Predict split...")
test_df, predict_df = train_test_split(temp_df, test_size=0.1, stratify=temp_df['diagnosis'], random_state=42)

# 4. Guardar CSVs con los archivos y etiquetas
print(f"Saving Train split...")
train_df.to_csv('data/train.csv', index=False)
print(f"Train samples: {len(train_df)}")

print(f"Saving Test split...")
test_df.to_csv('data/test.csv', index=False)
print(f"Test samples: {len(test_df)}")

print(f"Saving Predict split...")
predict_df.to_csv('data/predict.csv', index=False)
print(f"Predict samples: {len(predict_df)}")

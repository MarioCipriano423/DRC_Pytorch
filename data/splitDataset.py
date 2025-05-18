import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Cargar etiquetas originales
df = pd.read_csv('labels.csv')

# 2. Primer split: train (80%) y resto (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 3. Segundo split: test (10%) y predict (10%) a partir del resto
test_df, predict_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# 4. Guardar CSVs con los archivos y etiquetas
# Asumimos que las imágenes están en la carpeta 'colored_images/', así que guardamos solo el nombre para que la ruta quede clara al cargar
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
predict_df.to_csv('predict.csv', index=False)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Predict samples: {len(predict_df)}")

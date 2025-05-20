import pandas as pd
from sklearn.model_selection import train_test_split

print(f"Reading labels...")
LABELS_PATH = 'data/labels.csv'
df = pd.read_csv(LABELS_PATH)

print(f"Train split...")
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

print(f"Test-Predict split...")
test_df, predict_df = train_test_split(temp_df, test_size=0.1, stratify=temp_df['diagnosis'], random_state=42)

print(f"Saving Train split...")
train_df.to_csv('data/train.csv', index=False)
print(f"Train samples: {len(train_df)}")

print(f"Saving Test split...")
test_df.to_csv('data/test.csv', index=False)
print(f"Test samples: {len(test_df)}")

print(f"Saving Predict split...")
predict_df.to_csv('data/predict.csv', index=False)
print(f"Predict samples: {len(predict_df)}")

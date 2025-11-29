import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

CSV_PATH = r"C:\Users\Aishwarya S P\Downloads\UNSW_2018_IoT_Botnet_Dataset_71.csv"

# ==== 1. Load full dataset ====
df = pd.read_csv(CSV_PATH, low_memory=False)

label_col = "DDoS"   # this is the column with DDoS/Normal labels

# ==== 2. Separate attack + normal ====
df_attack = df[df[label_col] == "DDoS"]
df_normal = df[df[label_col] == "Normal"]

print("Original counts:")
print(df_attack.shape, "DDoS")
print(df_normal.shape, "Normal")

# ==== 3. Sample required rows ====
# 1000 attack (easy, dataset has many)
df_attack_1000 = df_attack.sample(1000, random_state=42)

# Oversample NORMAL (because dataset only has 28)
df_normal_oversampled = pd.concat([df_normal] * 40, ignore_index=True)

# Now sample exactly 1000 normal
df_normal_1000 = df_normal_oversampled.sample(1000, random_state=42)

# ==== 4. Merge balanced dataset ====
df_balanced = pd.concat([df_attack_1000, df_normal_1000], ignore_index=True)

print("\nBalanced 2000-row dataset:")
print(df_balanced[label_col].value_counts())

# ==== 5. Encode labels ====
le = LabelEncoder()
y_bal = le.fit_transform(df_balanced[label_col].astype(str))

# ==== 6. Preprocess features ====
X = df_balanced.drop(columns=[label_col])

for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].fillna("NA")
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    else:
        X[c] = X[c].fillna(0)

X = X.values.astype(np.float32)

# ==== 7. Scale ====
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==== 8. Train-test split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ==== 9. Build Neural Network ====
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.25),
    Dense(16, activation='relu'),
    Dropout(0.25),
    Dense(2, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("\nTraining Neural Network...")
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

# ==== 10. Evaluate model ====
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

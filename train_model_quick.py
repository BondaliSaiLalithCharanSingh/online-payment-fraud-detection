import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pickle

warnings.filterwarnings('ignore')

print("="*70)
print("ONLINE PAYMENTS FRAUD DETECTION - QUICK TRAINING")
print("="*70)

print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')
print(f"✓ Full dataset: {df.shape}")

# Use smaller sample for faster training
print("\n[2/7] Creating small sample (20,000 rows)...")
fraud_df = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0]

fraud_sample = fraud_df.sample(n=min(500, len(fraud_df)), random_state=42)
non_fraud_sample = non_fraud_df.sample(n=19500, random_state=42)

df = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Sample created: {df.shape}")

print("\n[3/7] Preprocessing...")
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
print(f"✓ Preprocessing complete")

print("\n[4/7] Splitting data...")
X = df.drop(['isFraud'], axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

print("\n[5/7] Training models (this will be quick)...\n")

results = {}

# Random Forest
print("  [1/5] Random Forest...", end=" ")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
results['Random Forest'] = rf_acc
print(f"✓ {rf_acc*100:.2f}%")

# Decision Tree
print("  [2/5] Decision Tree...", end=" ")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
results['Decision Tree'] = dt_acc
print(f"✓ {dt_acc*100:.2f}%")

# Extra Trees
print("  [3/5] Extra Trees...", end=" ")
et = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
et.fit(X_train, y_train)
et_acc = accuracy_score(y_test, et.predict(X_test))
results['Extra Trees'] = et_acc
print(f"✓ {et_acc*100:.2f}%")

# SVC with RBF kernel (faster than linear for small datasets)
print("  [4/5] SVC...", end=" ")
svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
svc_acc = accuracy_score(y_test, y_pred_svc)
results['SVC'] = svc_acc
print(f"✓ {svc_acc*100:.2f}%")

# XGBoost
print("  [5/5] XGBoost...", end=" ")
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
results['XGBoost'] = xgb_acc
print(f"✓ {xgb_acc*100:.2f}%")

print("\n[6/7] Model comparison...")
print("\n" + "="*70)
print("MODEL RESULTS")
print("="*70)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model_name = max(results, key=results.get)
print("\n" + "="*70)
print(f"BEST: {best_model_name} - {results[best_model_name]*100:.2f}%")
print("="*70)

print("\n[7/7] Saving model...")
with open('flask/payments.pkl', 'wb') as file:
    pickle.dump(svc, file)
print("✓ Model saved: flask/payments.pkl")

print("\nClassification Report (SVC):")
print("="*70)
print(classification_report(y_test, y_pred_svc, target_names=['Not Fraud', 'Is Fraud']))

cm = confusion_matrix(y_test, y_pred_svc)
print("\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0][0]:,}")
print(f"  False Positives: {cm[0][1]:,}")
print(f"  False Negatives: {cm[1][0]:,}")
print(f"  True Positives:  {cm[1][1]:,}")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  cd flask")
print("  python app.py")
print("  Open: http://127.0.0.1:5000/")
print("="*70)

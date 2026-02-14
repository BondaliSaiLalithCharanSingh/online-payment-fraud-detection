import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pickle

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

print("="*70)
print("ONLINE PAYMENTS FRAUD DETECTION - MODEL TRAINING")
print("="*70)

print("\n[1/8] Loading dataset...")
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

print("\n[2/8] Data preprocessing...")
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
print(f"✓ Unnecessary columns dropped")
print(f"  New shape: {df.shape}")

print("\n[3/8] Checking for null values...")
null_counts = df.isnull().sum()
print(f"✓ Null values check complete")
print(f"  Total null values: {null_counts.sum()}")

print("\n[4/8] Data statistics...")
print(f"✓ Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
print(f"  Legitimate transactions: {(df['isFraud']==0).sum()} ({(df['isFraud']==0).mean()*100:.2f}%)")

print("\n[5/8] Label encoding categorical features...")
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
print(f"✓ Label encoding completed")
print(f"  Transaction types encoded: {le.classes_}")

print("\n[6/8] Splitting data into train and test sets...")
X = df.drop(['isFraud'], axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Data split completed")
print(f"  Training set: {X_train.shape}")
print(f"  Testing set: {X_test.shape}")

print("\n[7/8] Training machine learning models...")
print("  This may take several minutes...\n")

results = {}

# Random Forest
print("  [1/5] Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = rf_acc
print(f"        ✓ Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# Decision Tree
print("  [2/5] Training Decision Tree Classifier...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
results['Decision Tree'] = dt_acc
print(f"        ✓ Accuracy: {dt_acc:.4f} ({dt_acc*100:.2f}%)")

# Extra Trees
print("  [3/5] Training Extra Trees Classifier...")
et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
y_pred_et = et.predict(X_test)
et_acc = accuracy_score(y_test, y_pred_et)
results['Extra Trees'] = et_acc
print(f"        ✓ Accuracy: {et_acc:.4f} ({et_acc*100:.2f}%)")

# Support Vector Classifier
print("  [4/5] Training Support Vector Classifier...")
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
svc_acc = accuracy_score(y_test, y_pred_svc)
results['SVC'] = svc_acc
print(f"        ✓ Accuracy: {svc_acc:.4f} ({svc_acc*100:.2f}%)")

# XGBoost
print("  [5/5] Training XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
results['XGBoost'] = xgb_acc
print(f"        ✓ Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

print("\n[8/8] Model comparison and saving...")
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model_name = max(results, key=results.get)
print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name} with {results[best_model_name]:.4f} accuracy")
print("="*70)

print("\nSaving the best model (SVC)...")
with open('flask/payments.pkl', 'wb') as file:
    pickle.dump(svc, file)
print("✓ Model saved successfully as 'flask/payments.pkl'")

print("\nDetailed Classification Report for SVC:")
print("="*70)
print(classification_report(y_test, y_pred_svc, target_names=['Not Fraud', 'Is Fraud']))

print("\nConfusion Matrix for SVC:")
print("="*70)
cm = confusion_matrix(y_test, y_pred_svc)
print(cm)
print(f"\nTrue Negatives:  {cm[0][0]:,}")
print(f"False Positives: {cm[0][1]:,}")
print(f"False Negatives: {cm[1][0]:,}")
print(f"True Positives:  {cm[1][1]:,}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Navigate to flask folder: cd flask")
print("2. Run the application: python app.py")
print("3. Open browser: http://127.0.0.1:5000/")
print("\n" + "="*70)

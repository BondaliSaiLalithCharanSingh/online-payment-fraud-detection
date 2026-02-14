# Dataset Folder

## Required Dataset

Place the following dataset file in this folder:

**Filename**: `PS_20174392719_1491204439457_log.csv`

**Source**: Kaggle - Online Payments Fraud Detection Dataset

**Download Link**: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset

## Dataset Information

- **Size**: ~470 MB (compressed), ~750 MB (uncompressed)
- **Rows**: 6,362,620 transactions
- **Columns**: 11 features

### Features Description

| Column | Description | Type |
|--------|-------------|------|
| step | Time unit (1 step = 1 hour) | Integer |
| type | Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) | String |
| amount | Transaction amount | Float |
| nameOrig | Customer ID who initiated the transaction | String |
| oldbalanceOrg | Initial balance before transaction (origin) | Float |
| newbalanceOrig | New balance after transaction (origin) | Float |
| nameDest | Customer ID who is the recipient | String |
| oldbalanceDest | Initial balance before transaction (destination) | Float |
| newbalanceDest | New balance after transaction (destination) | Float |
| isFraud | Target variable (1 = fraud, 0 = not fraud) | Integer |
| isFlaggedFraud | Flagged as fraud by system | Integer |

## Alternative Download Methods

### Method 1: Kaggle CLI
```bash
pip install kaggle
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset
```

### Method 2: KaggleHub (Python)
```python
import kagglehub
path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")
print("Path to dataset files:", path)
```

## Important Notes

1. The dataset file is **NOT included** in this repository due to its large size
2. You must download it separately from Kaggle
3. Ensure the filename matches exactly: `PS_20174392719_1491204439457_log.csv`
4. The file should be approximately 750 MB when uncompressed

## After Downloading

Once you've placed the dataset in this folder, you can proceed with:
1. Running the Jupyter notebook in `training/` folder
2. Training the machine learning models
3. Generating the `payments.pkl` model file

---

**Status**: ⚠️ Dataset Required - Please download before proceeding

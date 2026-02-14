# Quick Setup Guide - Online Payments Fraud Detection

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
Open Command Prompt or PowerShell and run:
```bash
cd "c:\Users\DINESH V A\Desktop\Online Fraud"
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Visit: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
2. Download: `PS_20174392719_1491204439457_log.csv`
3. Place in: `data/` folder

**Note**: The filename should be `PS_20174392719_1491204439457_log.csv` (without 's' at the end)

### Step 3: Train the Model
```bash
jupyter notebook
```
- Open: `training/ONLINE PAYMENTS FRAUD DETECTION.ipynb`
- Click: **Run All** (or Cell → Run All)
- Wait: ~5-10 minutes for training to complete
- Verify: `payments.pkl` is created in `flask/` folder

### Step 4: Run the Web Application
```bash
cd flask
python app.py
```

### Step 5: Access the Application
Open browser: http://127.0.0.1:5000/

---

## 📝 Sample Test Data

### Test Case 1: Fraudulent Transaction
- Step: `94`
- Type: `4` (TRANSFER)
- Amount: `14.590090`
- OldbalanceOrg: `2169679.91`
- NewbalanceOrig: `0.0`
- OldbalanceDest: `0.00`
- NewbalanceDest: `0.00`
- **Expected Result**: Is Fraud ⚠️

### Test Case 2: Legitimate Transaction
- Step: `1`
- Type: `3` (PAYMENT)
- Amount: `9.194174`
- OldbalanceOrg: `170136.00`
- NewbalanceOrig: `160236.36`
- OldbalanceDest: `0.00`
- NewbalanceDest: `0.00`
- **Expected Result**: Not Fraud ✅

---

## 🔢 Transaction Type Codes

| Code | Type |
|------|------|
| 0 | CASH_IN |
| 1 | CASH_OUT |
| 2 | DEBIT |
| 3 | PAYMENT |
| 4 | TRANSFER |

---

## ❗ Common Issues & Solutions

### Issue 1: "No module named 'sklearn'"
**Solution**:
```bash
pip install scikit-learn
```

### Issue 2: "FileNotFoundError: payments.pkl"
**Solution**: Run the Jupyter notebook first to train and save the model.

### Issue 3: "Dataset not found"
**Solution**: Ensure the CSV file is in the `data/` folder with the exact filename.

### Issue 4: Port 5000 already in use
**Solution**: Edit `app.py` and change:
```python
app.run(debug=False, port=5001)
```

### Issue 5: Jupyter notebook kernel dies
**Solution**: The dataset is large. Try:
- Close other applications
- Restart Jupyter
- Run cells one by one instead of "Run All"

---

## 🎯 Project Checklist

- [ ] Install Python 3.8+
- [ ] Install all dependencies
- [ ] Download dataset
- [ ] Place dataset in `data/` folder
- [ ] Run Jupyter notebook
- [ ] Verify `payments.pkl` exists
- [ ] Run Flask application
- [ ] Test predictions
- [ ] Review results

---

## 📊 Understanding the Results

### "Not Fraud" ✅
- Transaction appears legitimate
- Safe to proceed
- Low risk of fraudulent activity

### "Is Fraud" ⚠️
- Transaction shows suspicious patterns
- Requires manual review
- High risk of fraudulent activity

**Important**: Use ML predictions as one factor in decision-making, not the sole determinant.

---

## 🔧 Advanced Configuration

### Change Model Parameters
Edit the notebook and modify:
```python
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
```

### Adjust Training Data Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Enable Debug Mode
In `app.py`:
```python
app.run(debug=True)
```

---

## 📞 Need Help?

1. Check the main `README.md` for detailed documentation
2. Review the Jupyter notebook for training details
3. Verify all files are in correct locations
4. Ensure all dependencies are installed

---

**Happy Fraud Detection! 🔒**

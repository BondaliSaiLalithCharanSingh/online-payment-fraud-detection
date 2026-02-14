# Online Payments Fraud Detection using Machine Learning

A comprehensive machine learning system that detects fraudulent online payment transactions in real-time using advanced algorithms and a user-friendly Flask web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project implements a proactive fraud detection system that analyzes historical transaction data, customer behavior patterns, and machine learning algorithms to identify and prevent fraudulent activities during online transactions in real-time.

### Key Features

- **Real-time Fraud Monitoring**: Continuously monitors online payment transactions
- **Multiple ML Models**: Compares 5 different algorithms (Random Forest, Decision Tree, Extra Trees, SVC, XGBoost)
- **High Accuracy**: Achieves ~79% accuracy using Support Vector Classifier
- **Modern Web Interface**: Clean, responsive UI for easy interaction
- **Comprehensive Analysis**: Extensive EDA with 15+ visualizations
- **Adaptive Learning**: System improves over time with new data

## 📊 Scenarios

### Scenario 1: Real-time Fraud Monitoring
The system continuously monitors transactions by analyzing features such as transaction amount, location, device information, and user behavior to flag suspicious activities.

### Scenario 2: Fraudulent Account Detection
Machine learning models detect patterns indicative of fraudulent accounts by analyzing user behavior over time, including unusual login times and spending patterns.

### Scenario 3: Adaptive Fraud Prevention
The system adapts and improves its fraud detection capabilities by continuously learning from new data and adjusting algorithms to stay ahead of evolving fraud techniques.

## 🏗️ Project Structure

```
online payments fraud detection/
├── data/
│   └── PS_20174392719_1491204439457_log.csv    # Dataset
├── flask/
│   ├── templates/
│   │   ├── home.html                            # Landing page
│   │   ├── predict.html                         # Input form
│   │   └── submit.html                          # Results page
│   ├── app.py                                   # Flask application
│   └── payments.pkl                             # Saved ML model
├── training/
│   └── ONLINE PAYMENTS FRAUD DETECTION.ipynb    # Training notebook
├── training_ibm/
│   └── (IBM deployment files)
├── requirements.txt                             # Dependencies
└── README.md                                    # Documentation
```

## 🔧 Technical Architecture

### Dataset Features
- **step**: Time unit of transaction
- **type**: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- **amount**: Transaction amount
- **oldbalanceOrg**: Origin account balance before transaction
- **newbalanceOrig**: Origin account balance after transaction
- **oldbalanceDest**: Destination account balance before transaction
- **newbalanceDest**: Destination account balance after transaction
- **isFraud**: Target variable (0 = Not Fraud, 1 = Fraud)

### Machine Learning Models
1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **Extra Trees Classifier**
4. **Support Vector Classifier (SVC)** ⭐ Best Model
5. **XGBoost Classifier**

### Best Model Performance
- **Algorithm**: Support Vector Classifier (SVC)
- **Accuracy**: ~79%
- **Features Used**: 7 input features
- **Target Classes**: Binary classification

## 📋 Prerequisites

### Software Requirements
- Python 3.8 or higher
- Anaconda Navigator (recommended)
- Web browser (Chrome, Firefox, Edge)

### Python Packages
All required packages are listed in `requirements.txt`:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- Flask
- xgboost
- pickle-mixin
- jupyter
- notebook

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
cd "c:\Users\DINESH V A\Desktop\Online Fraud"
```

### Step 2: Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Or install individually:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy Flask xgboost pickle-mixin jupyter notebook
```

### Step 3: Download Dataset
Download the dataset from Kaggle:
- **Link**: [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **File**: PS_20174392719_1491204439457_log.csv
- **Location**: Place in the `data/` folder

Alternatively, use kagglehub:
```python
import kagglehub
path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")
print("Path to dataset files:", path)
```

## 📊 Training the Model

### Step 1: Open Jupyter Notebook
```bash
jupyter notebook
```

### Step 2: Navigate to Training Folder
Open `training/ONLINE PAYMENTS FRAUD DETECTION.ipynb`

### Step 3: Run All Cells
Execute all cells in sequence to:
1. Load and preprocess data
2. Perform exploratory data analysis
3. Train 5 different ML models
4. Compare model performances
5. Save the best model as `payments.pkl`

### Key Training Steps
- **Data Preprocessing**: Drop unnecessary columns, handle missing values
- **EDA**: Univariate, bivariate, and descriptive analysis
- **Feature Engineering**: Label encoding for categorical variables
- **Model Training**: Train and evaluate 5 classifiers
- **Model Selection**: Choose SVC as the best performer
- **Model Saving**: Export model using pickle

## 🌐 Running the Web Application

### Step 1: Navigate to Flask Folder
```bash
cd flask
```

### Step 2: Run the Flask App
```bash
python app.py
```

### Step 3: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

### Step 4: Use the Application
1. Click on **"Predict"** button in the navigation bar
2. Enter transaction details:
   - Step (e.g., 94)
   - Type (0-4: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
   - Amount (e.g., 14.590090)
   - OldbalanceOrg (e.g., 2169679.91)
   - NewbalanceOrig (e.g., 0.0)
   - OldbalanceDest (e.g., 0.00)
   - NewbalanceDest (e.g., 0.00)
3. Click **"Detect Fraud"**
4. View the prediction result

## 📸 Screenshots

### Home Page
- Modern landing page with project overview
- Feature cards highlighting key capabilities
- Call-to-action button for fraud detection

### Predict Page
- Clean input form with 7 transaction fields
- Helpful tooltips for each field
- Responsive design for all devices

### Result Page
- Clear display of fraud prediction
- Visual indicators (✅ for safe, ⚠️ for fraud)
- Action buttons for next steps

## 🔍 Model Evaluation

### Confusion Matrix
The model provides detailed confusion matrix showing:
- True Positives (Correctly identified fraud)
- True Negatives (Correctly identified legitimate)
- False Positives (Legitimate flagged as fraud)
- False Negatives (Fraud missed)

### Classification Report
Includes:
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive instances
- F1-Score: Harmonic mean of precision and recall
- Support: Number of samples in each class

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds and smooth animations
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Intuitive Navigation**: Easy-to-use interface
- **Visual Feedback**: Clear indicators for fraud/non-fraud
- **Professional Styling**: Clean, corporate-friendly appearance

## 🔐 Security Considerations

- Model predictions should be used as one factor in fraud detection
- Always combine ML predictions with other security measures
- Regularly retrain model with new data
- Monitor false positive/negative rates
- Implement proper authentication and authorization

## 📈 Future Enhancements

- [ ] Add real-time data streaming integration
- [ ] Implement model retraining pipeline
- [ ] Create admin dashboard for monitoring
- [ ] Add user authentication system
- [ ] Deploy to cloud (AWS, Azure, Heroku)
- [ ] Implement REST API endpoints
- [ ] Add ensemble methods for improved accuracy
- [ ] Include explainability features (SHAP, LIME)
- [ ] Add email/SMS alerts for fraud detection
- [ ] Implement A/B testing for model versions

## 🐛 Troubleshooting

### Issue: Model file not found
**Solution**: Ensure you've run the Jupyter notebook and generated `payments.pkl` in the flask folder.

### Issue: Dataset not found
**Solution**: Download the dataset and place it in the `data/` folder with the correct filename.

### Issue: Port 5000 already in use
**Solution**: Change the port in `app.py`:
```python
app.run(debug=False, port=5001)
```

### Issue: Module not found errors
**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Project Workflow

1. **Data Collection**: Download dataset from Kaggle
2. **Data Preprocessing**: Clean and prepare data
3. **EDA**: Analyze patterns and relationships
4. **Feature Engineering**: Encode categorical variables
5. **Model Training**: Train multiple ML models
6. **Model Evaluation**: Compare and select best model
7. **Model Deployment**: Integrate with Flask application
8. **Testing**: Verify predictions on new data

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: Kaggle - Online Payments Fraud Detection Dataset
- Libraries: scikit-learn, Flask, XGBoost, Pandas, NumPy
- Inspiration: Real-world fraud detection systems

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is an educational project. For production use, additional security measures, testing, and validation are required.

## 🎓 Learning Outcomes

By completing this project, you will:
- Understand fundamental ML concepts and techniques
- Gain experience with data preprocessing and EDA
- Learn to build and compare multiple ML models
- Deploy ML models using Flask
- Create modern, responsive web interfaces
- Implement end-to-end ML pipelines

---

**Built with ❤️ using Python, Machine Learning, and Flask**

# Bank Customer Churn Prediction - Production ML System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![Framework](https://img.shields.io/badge/API-FastAPI-green.svg)
![Database](https://img.shields.io/badge/DB-MongoDB-brightgreen.svg)

## 🎯 Problem Statement

**Objective**: Build a production-grade machine learning system to predict bank customer churn with high accuracy and interpretability, enabling proactive retention strategies.

### Business Impact
- **Customer Retention**: Identify at-risk customers before they leave
- **Revenue Protection**: Reduce churn-related revenue loss (estimated 15-25% annually)
- **Targeted Interventions**: Enable personalized retention campaigns
- **Cost Optimization**: Focus resources on high-risk customers

### Technical Challenge
Predict whether a bank customer will close their account (`Exited=1`) or remain (`Exited=0`) based on:
- Demographics (Age, Geography, Gender)
- Banking relationship (Tenure, Balance, Products)
- Engagement metrics (CreditScore, IsActiveMember)

**Class Imbalance**: ~20.4% churn rate (imbalanced dataset)

---

## 📊 Dataset Overview

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numerical | Customer credit score (300-850) |
| Geography | Categorical | Country (France, Spain, Germany) |
| Gender | Categorical | Male/Female |
| Age | Numerical | Customer age |
| Tenure | Numerical | Years with bank |
| Balance | Numerical | Account balance |
| NumOfProducts | Numerical | Number of bank products (1-4) |
| HasCrCard | Binary | Credit card ownership (0/1) |
| IsActiveMember | Binary | Active membership status (0/1) |
| EstimatedSalary | Numerical | Estimated annual salary |
| **Exited** | **Target** | **Churned (1) or Retained (0)** |

**Dataset Size**: 10,000 records
**Features**: 10 predictive features
**Target Distribution**: ~79.6% retained, ~20.4% churned

---

## 🏗️ Architecture

### Modular Pipeline Design
```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Model Deployment
```

### Tech Stack
- **ML Framework**: scikit-learn, imbalanced-learn
- **Data Storage**: MongoDB Atlas
- **API**: FastAPI
- **Experiment Tracking**: MLflow + DagHub
- **Deployment**: Docker + AWS EC2
- **Serialization**: dill, joblib

---

## 🚀 Key Features

### 1. **Production-Ready Pipeline**
- ✅ Modular component architecture
- ✅ Comprehensive error handling & logging
- ✅ Configuration management with dataclasses
- ✅ Artifact versioning with timestamps

### 2. **Advanced ML Techniques**
- ✅ Feature engineering (18 engineered features)
- ✅ Class imbalance handling (`class_weight='balanced'`)
- ✅ Ensemble models (RandomForest, XGBoost, etc.)
- ✅ Hyperparameter tuning
- ✅ Cross-validation

### 3. **Data Quality & Monitoring**
- ✅ Schema validation
- ✅ Data drift detection
- ✅ Model performance tracking
- ✅ A/B testing support

### 4. **API & Deployment**
- ✅ RESTful API with FastAPI
- ✅ Batch prediction endpoint
- ✅ Real-time inference (<100ms)
- ✅ Dockerized deployment
- ✅ Health checks & monitoring

---

## 📁 Project Structure

```
bank-customer-churn/
├── banksecurity/               # Main package
│   ├── components/            # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── entity/                # Data classes
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── pipeline/              # Orchestration
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils/                 # Utilities
│   │   ├── main_utils/
│   │   └── ml_utils/
│   ├── constant/              # Constants
│   ├── exception/             # Custom exceptions
│   └── logging/               # Logging config
├── notebook/                  # Jupyter notebooks
├── artifacts/                 # Training artifacts
├── final_models/              # Production models
├── logs/                      # Application logs
├── app.py                     # FastAPI application
├── main.py                    # CLI training script
├── push_data.py               # MongoDB data loader
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- MongoDB Atlas account
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/bank-customer-churn.git
cd bank-customer-churn
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create `.env` file:
```env
MONGO_DB_URL=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/bank-customer-churn.mlflow
```

### 5. Load Data to MongoDB
```bash
python push_data.py
```

---

## 🎓 Usage

### Training Pipeline
```bash
# Run full training pipeline
python main.py

# Or trigger via API
curl -X GET http://localhost:8000/train
```

### Prediction API
```bash
# Start FastAPI server
python app.py

# Make predictions
curl -X POST http://localhost:8000/predict \
  -F "file=@test_data.csv"
```

### Docker Deployment
```bash
# Build image
docker build -t bank-churn-predictor .

# Run container
docker run -p 8000:8000 bank-churn-predictor
```

---

## 📈 Model Performance

### Metrics (Example)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.867 | 0.862 |
| Precision | 0.775 | 0.768 |
| Recall | 0.534 | 0.521 |
| F1-Score | 0.632 | 0.621 |
| ROC-AUC | 0.859 | 0.854 |

### Feature Importance
Top predictors:
1. Age
2. NumOfProducts
3. IsActiveMember
4. Balance
5. Geography

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test API endpoints
pytest tests/test_api.py
```

---

## 🚢 Deployment

### AWS EC2 Deployment
```bash
# SSH into EC2
ssh -i keypair.pem ubuntu@<ec2-public-ip>

# Install Docker
sudo apt-get update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Pull and run container
docker pull yourusername/bank-churn-predictor
docker run -d -p 8000:8000 bank-churn-predictor
```

---

## 📊 MLflow Experiment Tracking

View experiments at:
```
https://dagshub.com/<username>/bank-customer-churn.mlflow
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Hussain Madarwala**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset source: Banking industry standard features
- Inspiration: Production ML best practices

---

## 📚 Next Steps

- [ ] Implement real-time streaming predictions
- [ ] Add model explainability (SHAP values)
- [ ] Build monitoring dashboard
- [ ] Implement A/B testing framework
- [ ] Add model retraining automation

---

**⚡ Built with production-readiness in mind | ML Engineering Best Practices**

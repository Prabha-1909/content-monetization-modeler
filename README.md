# Content Monetization Modeler

## 📌 Project Overview

The **Content Monetization Modeler** is a machine learning project that predicts **YouTube ad revenue** based on various video performance metrics such as views, engagement, and audience behavior.

This project demonstrates the complete **data science workflow**, including:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model training
* Model evaluation
* Deployment using Streamlit

The final model predicts potential ad revenue for content creators and media companies.

---

## 🎯 Problem Statement

With the rapid growth of content platforms like YouTube, creators and companies rely heavily on ad revenue for monetization. Predicting revenue based on video performance metrics can help creators understand how their content might perform financially.

This project builds a machine learning model to estimate **expected YouTube ad revenue** from available video analytics.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## Project Structure

```

Content-Monetization-Modeler

app
└── streamlit_app.py          # Streamlit web application

data
├── youtube_ad_revenue_dataset.csv
├── cleaned_youtube_data.csv
└── featured_youtube_data.csv

models
└── revenue_model.pkl         # Trained ML model

src
├── data_preprocessing.py
├── eda_analysis.py
├── feature_engineering.py
├── train_model.py
└── evaluate_model.py

requirements.txt
.gitignore
README.md
```

---

## ⚙️ Machine Learning Models Used

The following regression models were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

### Model Performance

| Model             | R² Score | MAE  | RMSE  |
| ----------------- | -------- | ---- | ----- |
| Linear Regression | 0.952    | 3.08 | 13.47 |
| Decision Tree     | 0.898    | 5.39 | 19.72 |
| Random Forest     | 0.949    | 3.58 | 13.88 |
| Gradient Boosting | 0.952    | 3.62 | 13.53 |

✅ **Best Model:** Linear Regression

The trained model is saved as:

models/revenue_model.pkl

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/Content-Monetization-Modeler.git

cd Content-Monetization-Modeler

---

### 2️⃣ Install Dependencies

pip install -r requirements.txt

---

### 3️⃣ Run Model Training

python src/train_model.py

---

### 4️⃣ Evaluate Model

python src/evaluate_model.py

---

### 5️⃣ Run Streamlit Application

streamlit run app/streamlit_app.py

---

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Deployment with Streamlit

---

## 📈 Key Learnings

* Data preprocessing techniques
* Feature engineering for predictive modeling
* Training and evaluating regression models
* Model comparison using performance metrics
* Deploying machine learning models using Streamlit

---

## 🔮 Future Improvements

* Add more advanced machine learning models
* Hyperparameter tuning
* Use real YouTube API data
* Deploy the application on a cloud platform

---



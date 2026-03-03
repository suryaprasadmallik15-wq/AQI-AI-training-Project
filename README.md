# AQI-AI-training-Project

🌍 Air Quality Index (AQI) Prediction using Machine Learning
📌 Project Overview
This project predicts Air Quality Index (AQI) values using Machine Learning algorithms based on environmental pollutant data.

The model is trained using the city_day.csv dataset and deployed with Streamlit to provide an interactive web interface for AQI prediction.

🚀 Features
Data preprocessing and cleaning
Exploratory Data Analysis (EDA)
Multiple regression algorithms
Model performance comparison
Interactive Streamlit frontend
Real-time AQI prediction
🧠 Machine Learning Algorithms Used
Linear Regression
K-Nearest Neighbors (KNN) Regressor
Decision Tree Regressor
Random Forest Regressor
📂 Dataset
Dataset Used: city_day.csv

Features include:

PM2.5
PM10
NO2
SO2
CO
O3
Temperature
Humidity
Wind Speed
AQI (Target Variable)
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Streamlit
📊 Model Evaluation Metrics
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
📁 Project Structure
AQI-PREDECTION/
│
├── AIR_QUALITY_PREDECTION.ipynb
├── app.py
├── city_day.csv
├── requirements.txt
└── README.md
▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run Streamlit App
streamlit run app.py
Then open:

http://localhost:8501
⚙️ Workflow
Load dataset (city_day.csv)
Data preprocessing
Train/Test split
Train multiple ML models
Evaluate models
Deploy with Streamlit
Predict AQI based on user inputs
📈 Future Improvements
Add XGBoost / Gradient Boosting
Deploy to cloud (Render / AWS)
Add AQI category classification
Improve UI/UX design
🎯 Learning Outcomes
End-to-end Machine Learning pipeline
Regression model implementation
Model evaluation techniques
Streamlit deployment
Real-world environmental data analysis
👨‍💻 Author
Your's Truly Ranjan
B.Tech – AI & Data Science
Machine Learning Enthusiast

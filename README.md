# 📈 Stock Price Prediction using LSTM and Random Forest

This project is a comprehensive application of data science and machine learning techniques to predict Apple Inc. (AAPL) stock prices using historical financial data. It includes the full data science pipeline: data collection, cleaning, exploration, visualization, feature engineering, and predictive modeling using both LSTM (deep learning) and Random Forest (ensemble method).

---

## 🧠 Project Objectives

- Predict AAPL stock prices using real-world historical data
- Compare performance between LSTM and Random Forest models
- Gain insights into stock price patterns using advanced visualizations
- Apply time series techniques for sequential financial data

---

## 📊 Technologies & Libraries Used

- **Python** (Jupyter Notebook)
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Plotly**
- **Scikit-learn**, **TensorFlow/Keras**
- **Yahoo Finance API** (`yfinance`)
- **LSTM (Long Short-Term Memory Networks)**
- **Random Forest Regressor**

---
## Machine Learning Pipeline in detail:

Data Collection: Pulled historical stock price data from Yahoo Finance.
Data Cleaning: Handled missing values and formatted the dataset.
EDA & Visualization: Time series plots, boxplots, correlation heatmaps, and candlestick charts.
Modeling:
LSTM: Built and trained deep learning model for sequential forecasting
Random Forest: Applied feature engineering and regression
Model Evaluation:
RMSE, MAE, R² Score
LSTM outperformed Random Forest in accuracy but required more training time.

---

## 📁 Project Structure

```bash
stock-price-prediction/
├── stock_price_prediction.ipynb     # Main Jupyter Notebook
├── README.md                        # Project Documentation
└── presentation/
    └── Data-Science-and-Machine-Learning-in-Action.pptx

---

## How to run

cloning repo into your local: git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
install dependencies: pip install -r requirements.txt
run the jupyter notebook stock_price_prediction.ipynb

---







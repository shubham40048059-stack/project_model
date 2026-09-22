# House Price Prediction Project

This project builds a machine learning model to predict house prices using property-related features such as square footage, number of bedrooms, bathrooms, lot size, garage size, year built, and neighborhood quality.

## Dataset

The dataset used in this project is included in the workspace as:
- [house_price.csv](house_price.csv)

It contains 1,000 records and 8 columns, including the target variable `House_Price`.

## Project Description

The goal of this project is to:
- analyze the housing dataset,
- prepare the features for modeling,
- train a regression model,
- evaluate predictive performance,
- estimate house prices for unseen data.

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Setup Instructions

1. Open the project folder.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the notebook or script:
   ```bash
   jupyter notebook Student_HousePricePrediction.ipynb
   ```
   or
   ```bash
   python Student_HousePricePrediction.py
   ```

## Key Information

- Target variable: `House_Price`
- Model used: `RandomForestRegressor`
- Train/test split: 80/20
- Evaluation metrics: MAE, MSE, RMSE, and R2 score

## Notes

This project is designed for educational and demonstration purposes, and it uses a structured synthetic housing dataset that is suitable for regression modeling exercises.

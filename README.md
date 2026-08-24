# Stock Price Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict stock prices based on historical stock market data.

Historical data of Apple Inc. (AAPL) is collected using Yahoo Finance. The previous 60 days of closing prices are used as input features to predict the next day's closing price.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Yahoo Finance
- Google Colab

## Machine Learning Model

The project uses **Linear Regression** for stock price prediction.

### Input

Previous 60 days of AAPL closing prices.

### Output

Predicted next-day closing price.

## Model Evaluation

The model is evaluated using:

- MAE: 2.56
- RMSE: 3.66
- MAPE: 1.20%

These values represent the model's prediction error on the test dataset.

## Project Workflow

1. Collect historical stock data
2. Preprocess the data
3. Create features using the previous 60 days
4. Split the data into training and testing sets
5. Train the Linear Regression model
6. Generate predictions
7. Evaluate the model
8. Visualize actual vs predicted prices
9. Predict the next-day closing price

## Result

For the current trained model, the predicted next-day closing price was:

**$271.18**

This is a machine-learning estimate and is not a guaranteed future market price.

## Disclaimer

This project is developed for educational and academic purposes. Stock-market predictions are uncertain and should not be considered financial advice.

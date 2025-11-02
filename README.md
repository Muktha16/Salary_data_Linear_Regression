Project Overview

This project demonstrates how Linear Regression can be used to predict salaries based on years of experience.
It’s a simple yet powerful example of Supervised Machine Learning, where we train a model to learn the relationship between two variables — Years of Experience (independent variable) and Salary (dependent variable).

Objective

To build a machine learning model that accurately predicts an employee’s salary based on their years of experience using Linear Regression.

Dataset Information

Dataset Name: Salary Dataset
File Name: Salary_dataset.csv

Column Name	Description	Data Type
YearsExperience	Number of years of professional experience	float
Salary	Annual salary (in USD)	float

Dataset Size: 30 rows × 2 columns
Source: Public dataset commonly used for regression practice (can be from Kaggle)

Steps in the Project

1.Import Libraries
Load essential Python libraries like pandas, numpy, matplotlib, and scikit-learn.

2.Load the Dataset
Use pandas to read the Salary_Data.csv file.

3.Data Exploration

View the first few rows using df.head().

Check for missing values and data types.

Visualize the data using a scatter plot (YearsExperience vs. Salary).

4.Data Splitting
Split the dataset into:

Training Set: 80%

Testing Set: 20%

Using:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


5.Model Training
Fit a Linear Regression model:

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


6.Prediction
Predict salaries using:

y_pred = model.predict(X_test)


7.Model Evaluation
Calculate metrics such as:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

8.Visualization
Plot:

Training data with regression line

Test data with predicted results

Technologies Used

Tool	Purpose
🐍 Python	Programming Language
📦 Pandas	Data manipulation
🔢 NumPy	Numerical computation
📊 Matplotlib / Seaborn	Data visualization
🤖 Scikit-Learn	Machine Learning

Project Structure
Salary_Prediction_Linear_Regression/
│
├── Salary_Data.csv
├── salary_prediction.ipynb
├── README.md

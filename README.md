# Machine Learning Evaluation App

This is an interactive web app built using Streamlit that allows users to upload datasets, select features, choose machine learning models, and evaluate their performance. The app supports both classification and regression tasks with various machine learning models.

## Features

- **Dataset Upload**: Upload CSV or Excel files.
- **Missing Value Handling**: Option to handle missing values by imputing the most frequent value.
- **Model Selection**: Choose from multiple machine learning models for classification and regression tasks.
- **Data Encoding**: Automatically encodes categorical data for model compatibility.
- **Evaluation Metrics**: Displays various evaluation metrics for the selected model.
  - For classification: Classification report, confusion matrix, ROC curve.
  - For regression: Mean squared error, R-squared, and residual plots.
- **Feature Importance**: Displays the importance of features for tree-based models (Random Forest, Decision Tree).

## Technologies Used

- **Streamlit**: To create the interactive web app.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-Learn**: For machine learning models, metrics, and preprocessing.
- **Matplotlib & Seaborn**: For visualizing the data and evaluation results.

## Installation

To run this project locally, you need to install the required dependencies. Follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/abdulrehman898998/ml_evaluation_model.git

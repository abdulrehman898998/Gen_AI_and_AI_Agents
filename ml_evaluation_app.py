import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# App title
st.title("Interactive Machine Learning Evaluation App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(df.sample(10))

    # Handle missing values
    if st.checkbox("Handle missing values"):
        imputer = SimpleImputer(strategy="most_frequent")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        st.write("Missing values handled. Updated dataset:")
        st.dataframe(df.sample(10))

    # Feature and target selection
    features = st.multiselect("Select features", options=df.columns)
    target = st.selectbox("Select target variable", options=df.columns)

    if features and target:
        # Select task type
        task = st.radio("Select task type", ["Classification", "Regression"])

        # Encode categorical data for selected features and target
        st.write("Encoding categorical data...")
        for col in features + [target]:
            if df[col].dtype == "object":
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                st.write(f"Encoded column: {col}")

        X = df[features].values
        y = df[target].values

        # Display class distribution for classification
        if task == "Classification":
            class_counts = Counter(y)
            st.write("Class Distribution:", class_counts)

            # Check for sparse classes
            min_samples = 2
            sparse_classes = [cls for cls, count in class_counts.items() if count < min_samples]
            if sparse_classes:
                st.warning(f"The following classes have fewer than {min_samples} samples: {sparse_classes}. These classes will be removed.")
                mask = ~np.isin(y, sparse_classes)
                X = X[mask]
                y = y[mask]
                class_counts = Counter(y)
                st.write("Updated Class Distribution:", class_counts)

        # Split the data
        try:
            stratify = y if task == "Classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            st.write(f"Train-Test Split: {len(X_train)} training samples, {len(X_test)} testing samples.")
        except ValueError as ve:
            st.error(f"Error during data splitting: {ve}")
            X_train, X_test, y_train, y_test = None, None, None, None  # Ensure variables are defined

        # Train and evaluate the model if splitting was successful
        if X_train is not None and y_train is not None:
            if task == "Classification":
                model_type = st.selectbox("Select Classification Model", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM", "KNN", "Naive Bayes"])
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                elif model_type == "Decision Tree":
                    max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                elif model_type == "SVM":
                    model = SVC(probability=True, random_state=42)
                elif model_type == "KNN":
                    n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                elif model_type == "Naive Bayes":
                    model = GaussianNB()

                # Train model with epochs
                epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=10)
                for epoch in range(epochs):
                    model.fit(X_train, y_train)
                    st.write(f"Epoch {epoch + 1}/{epochs} completed.")

                # Predictions
                predictions = model.predict(X_test)

                # Classification Metrics
                st.write("Classification Report:")
                report = classification_report(y_test, predictions, output_dict=True, target_names=df[target].unique().astype(str))
                st.dataframe(pd.DataFrame(report).transpose())

                # Confusion Matrix
                cm = confusion_matrix(y_test, predictions)
                st.write("Confusion Matrix:")
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
                st.pyplot(plt)

                # ROC Curve
                if len(np.unique(y)) == 2:  # Binary classification
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
                    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver Operating Characteristic")
                    plt.legend(loc="lower right")
                    st.pyplot(plt)

            elif task == "Regression":
                model_type = st.selectbox("Select Regression Model", ["Random Forest", "Linear Regression", "Decision Tree", "KNN"])
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                elif model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Decision Tree":
                    max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif model_type == "KNN":
                    n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)

                # Train model
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Regression Metrics
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Root Mean Squared Error: {rmse:.4f}")
                st.write(f"R-squared: {r2:.4f}")

                # Actual vs Predicted Plot
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, predictions, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Actual vs Predicted")
                st.pyplot(plt)

                # Residual Plot
                residuals = y_test - predictions
                plt.figure(figsize=(8, 6))
                sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={"color": "red"})
                plt.xlabel("Predicted")
                plt.ylabel("Residuals")
                plt.title("Residuals vs Predicted")
                st.pyplot(plt)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.title("📊 Auto ML Data Analyzer")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📉 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Statistical Summary")
    st.write(df.describe(include='all'))

    st.subheader("📈 Histograms for Numerical Columns")
    for col in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    num_cols = df.select_dtypes(include=[np.number])
    if num_cols.shape[1] >= 2:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Pairplot
    if num_cols.shape[1] >= 2:
        st.subheader("🔗 Pairplot")
        fig = sns.pairplot(num_cols)
        st.pyplot(fig.figure)

    # Target column input via dropdown
    st.subheader("🎯 Select Target Column")
    y_col = st.selectbox("Select target column (Y)", df.columns)

    if y_col:
        df = df.dropna(subset=[y_col])  # Drop rows with missing target values
        df = df.dropna()  # Drop any remaining rows with missing values

        X = df.drop(columns=[y_col])
        y = df[y_col]

        # Handle categorical features automatically
        X = pd.get_dummies(X, drop_first=True)

        # Determine if target is numeric or categorical
        target_is_numeric = pd.api.types.is_numeric_dtype(y)

        if not target_is_numeric and len(y.unique()) <= 10:
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_is_numeric = True

        if len(X) != len(y):
            st.error("Mismatch in number of rows between features and target.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if target_is_numeric and len(np.unique(y)) > 10:  # Regression
                st.subheader("📘 Regression Models")

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                preds = lr.predict(X_test)
                st.write("📌 **Linear Regression**")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
                st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")

                dt = DecisionTreeRegressor()
                dt.fit(X_train, y_train)
                preds_dt = dt.predict(X_test)
                st.write("📌 **Decision Tree Regressor**")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds_dt)):.4f}")
                st.write(f"R2 Score: {r2_score(y_test, preds_dt):.4f}")

                rf = RandomForestRegressor()
                rf.fit(X_train, y_train)
                preds_rf = rf.predict(X_test)
                st.write("📌 **Random Forest Regressor**")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds_rf)):.4f}")
                st.write(f"R2 Score: {r2_score(y_test, preds_rf):.4f}")

            else:  # Classification
                st.subheader("📕 Classification Models")

                def model_eval(model, name):
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    st.write(f"📌 **{name}**")
                    st.write("Accuracy:", accuracy_score(y_test, preds))
                    st.text(classification_report(y_test, preds))

                if len(np.unique(y)) > 1:
                    model_eval(LogisticRegression(max_iter=1000), "Logistic Regression")
                    model_eval(DecisionTreeClassifier(), "Decision Tree Classifier")
                    model_eval(RandomForestClassifier(), "Random Forest Classifier")
                    model_eval(KNeighborsClassifier(), "K-Nearest Neighbors")
                else:
                    st.warning("Only one class present in target. Skipping classification models.")

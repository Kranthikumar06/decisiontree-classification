import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
st.title("🌸 Iris Flower Classification with Decision Tree")
st.write("This app uses a Decision Tree Classifier to predict the species of an iris flower based on its measurements.")

# Load and prepare data
@st.cache_resource
def load_and_train_model():
    """Load iris dataset and train the decision tree model"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, iris, X, y

# Load model and data
model, iris, X, y = load_and_train_model()

# Prediction Interface
st.header("Make a Prediction")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.5,
        step=0.1
    )
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=3.5,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1
    )
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1
    )

# Make prediction
if st.button("🎯 Predict", type="primary"):
    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    })
    
    prediction = model.predict(input_data)[0]
    predicted_species = iris.target_names[prediction]
    
    st.success(f"## 🌸 Predicted Species: **{predicted_species.capitalize()}**")
    
    # Display input summary
    st.write("### 📋 Input Summary:")
    summary_df = pd.DataFrame({
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(summary_df, use_container_width=True)


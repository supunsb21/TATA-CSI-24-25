import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# Set page layout and style
st.set_page_config(page_title="TATA Commercial CSI 24/25", page_icon="📊", layout="wide")

# Apply a clean, minimal style for the app
st.markdown("""
    <style>
    .css-1d391kg {font-family: 'Roboto', sans-serif;}
    .stSelectbox, .stButton, .stTextInput {font-size: 16px;}
    .stMarkdown {padding-top: 0.8rem; padding-bottom: 1.2rem;}
    .css-ffhzg2 {background-color: #f0f4f8;}
    </style>
""", unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv("df_cleaned.csv")

# Select target and features
X = df.iloc[:, 0:7]
y = df['Overall Evaluation']
target = 'Overall Evaluation'
features = X.columns

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox('Choose Regression Model', ['Linear Regression', 'Elastic Net Regression'])

# Load the selected model with error handling
try:
    if model_choice == 'Linear Regression':
        model = joblib.load('LinearReg.joblib')
    elif model_choice == 'Elastic Net Regression':
        model = joblib.load('ElasticNet.joblib')
except FileNotFoundError:
    st.error(f"Model file for {model_choice} not found. Please check the path.")
    st.stop()

# Predict with the model
y_pred = model.predict(X)

st.header('TATA Commercial CSI 2024/2025')

# Section: Dataset Preview
with st.expander("📋 Preview Dataset", expanded=False):
    st.dataframe(df.head(10))

# Section: Target Variable and Features
with st.expander("🎯 Target Variable & Features", expanded=True):
    st.header("Target Variable and Features")
    st.markdown(f"**Target Variable**: {target}")
    st.markdown(f"**Features**: {', '.join(features)}")

# Section: Model Evaluation
num_data_points = len(df)
with st.expander("📈 Model Evaluation", expanded=True):
    st.header(f'Model Evaluation: {model_choice}')
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.markdown(f'### Number of Surveys: {num_data_points}')
    st.markdown(f"### Mean Squared Error (MSE): {mse:.2f}")
    st.markdown(f'### R²: {r2 * 100:.2f}% Variation in Overall Evaluation Can Be Explained')

# Model Equation 
with st.expander("Model Equation", expanded=True):
    st.header('Model Equation')
    coefficients = model.coef_
    intercept = model.intercept_

    # Create the equation string with colored text for coefficients and feature names
    equation_html = f'<p><strong>{target} = <span style="color: #FFFFFF;">{intercept:.2f}</span></strong>'

    for i, coef in enumerate(coefficients):
        coef_color = "#1E90FF" if coef >= 0 else "#FF4500"  # Blue for positive, red for negative coefficients
        equation_html += f' <span style="color: {coef_color};">{"+" if coef >= 0 else "-"} {abs(coef):.2f}</span> * <span style="color: #32CD32;">{features[i]}</span>'

    equation_html += '</p>'

    # Display the equation using HTML for better clarity and color formatting
    st.markdown(equation_html, unsafe_allow_html=True)

# Section: Feature Importance Plot
with st.expander("📊 Feature Importance Plot", expanded=True):
    st.header('Feature Importance Plot')
    feature_importance = coefficients
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    })
    importance_df['Absolute Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Absolute Importance', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Importance'],
        orientation='h',
        text=importance_df['Importance'].apply(lambda x: f'{x:.2f}'),
        textposition='outside',
        marker=dict(color=importance_df['Importance'], colorscale='RdBu', showscale=True, line=dict(width=1, color='black'))
    ))

    fig.update_layout(
        title='Feature Importance in Regression with Coefficients (+/-)',
        xaxis_title='Importance (Coefficient Value)',
        yaxis_title='Feature',
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(family="Roboto, sans-serif", size=14, color="white"),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


# Section: Real-Time Prediction from User Input (Discrete Values)
with st.expander("🔍 Predict from Custom CSI Values", expanded=True):
    st.subheader("🧾 Input CSI Scores (0 to 10)")
    st.markdown("""
        Adjust the CSI scores below to see how they affect the predicted **Overall Evaluation**.
        Each score represents customer satisfaction in a specific area.
    """)

    # Create a two-column layout for better visual structure
    col1, col2 = st.columns(2)
    user_input = []

    # Distribute feature inputs across two columns
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            value = st.selectbox(
                f"{feature}",
                options=list(range(11)),
                index=5,
                key=f"user_input_{feature}"
            )
            user_input.append(value)

    # Predict using the model in real-time
    input_df = pd.DataFrame([user_input], columns=features)
    prediction = model.predict(input_df)[0]

    # Display prediction result with styling
    st.markdown("---")
    st.markdown("### 📈 Predicted Overall Evaluation")
    st.metric(label="Predicted Score", value=f"{prediction:.2f}")
    st.markdown("This score reflects the expected overall customer satisfaction based on the CSI inputs.")


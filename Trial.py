import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="LendSmart Dashboard", page_icon="LendSmart_logo.png")

# Load the dataset
data = pd.read_csv('loan_balanced_6040.csv')

# Custom CSS
st.markdown("""
    <style>
    .title-blue {
        color: #1B49A4;
    }
    .sidebar .sidebar-content {
        padding-top: 0;
    }
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;  /* Ensure width is not too large */
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1B49A4 !important;
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] .css-1wy0on6 {
        background-color: #1B49A4 !important;
    }
    .stSelectbox div[data-baseweb="select"] .css-1hb7zxy-IndicatorsContainer {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] .css-1uccc91-singleValue {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] .css-1n7v3ny-option {
        color: black !important;
    }
    .stButton button {
        background-color: #1B49A4 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the logo in the sidebar
try:
    st.sidebar.image("LendSmart_logo.png", use_column_width=True)
except Exception as e:
    st.sidebar.write("Logo not found. Please ensure 'LendSmart_logo.png' is in the correct directory.")

# Display the logo at the top of the main page
try:
    st.markdown('<div style="text-align: center;"><img src="LendSmart_logo.png" class="logo"></div>', unsafe_allow_html=True)
except Exception as e:
    st.write("Logo not found. Please ensure 'LendSmart_logo.png' is in the correct directory.")

# Data preprocessing
X = data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']]
y = data['loan_status']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define the Random Forest Classifier with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt'],
    'max_depth': [4, 6],
    'criterion': ['gini']
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Prepare data for linear regression to predict interest rates
X_interest = data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
y_interest = data['int_rate']

# Train a Linear Regression model for predicting interest rates
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_interest, y_interest)

# Compute default probabilities for each client using the Random Forest Classifier
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:, 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Define the layout of the app
st.title("Loan Default Prediction Dashboard")
st.sidebar.title("Navigation")
tabs = ["Main Page", "Background Information", "New Client Default Prediction", "Client Risk Segmentation"]
selected_tab = st.sidebar.radio("Tabs", tabs)

if selected_tab == "Main Page":
    st.markdown("<h1 class='title-blue'>Loan Default Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.write("""
    This dashboard helps a US loan mortgage company identify and manage at-risk clients. Using machine learning models and statistical analysis, it predicts loan defaults and provides actionable insights. Amid rising US mortgage delinquency rates due to economic uncertainty (Financial Times), this tool enables early identification of potential defaults and better management of at-risk clients, ensuring financial stability and improved loan portfolio management.
    """)

elif selected_tab == "Background Information":
    st.markdown("<h1 class='title-blue'>Background Information</h1>", unsafe_allow_html=True)
    st.write("""
    Explore various graphs that describe our dataset, which underpins the predictive tools used in the following tabs. Gain insights into loan distributions, income levels, interest rates, and more.
    """)

    dropdown_selection = st.selectbox("Select a graph", ["Correlation Heatmap", "Distribution of Loan Status", "Distribution of Loan Amounts", "Distribution of Annual Incomes", "Distribution of Interest Rates"])

    if dropdown_selection == "Correlation Heatmap":
        correlation_matrix = data[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'delinq_2yrs', 'home_ownership_OWN', 'open_acc', 'loan_status']].corr()
        fig = px.imshow(correlation_matrix, labels={'color': 'Correlation'}, color_continuous_scale='RdBu_r')
        fig.update_layout(title='Correlation Heatmap')
        st.plotly_chart(fig)
    elif dropdown_selection == "Distribution of Loan Status":
        loan_status_counts = data['loan_status'].value_counts().reset_index()
        loan_status_counts.columns = ['Loan Status', 'Count']
        fig = px.bar(loan_status_counts, x='Loan Status', y='Count', labels={'Loan Status': 'Loan Status', 'Count': 'Number of Loans'}, title='Distribution of Loan Status')
        st.plotly_chart(fig)
    elif dropdown_selection == "Distribution of Loan Amounts":
        fig = px.histogram(data, x='loan_amnt', nbins=30, title='Distribution of Loan Amounts')
        fig.update_layout(xaxis_title='Loan Amount ($)', yaxis_title='Count')
        st.plotly_chart(fig)
    elif dropdown_selection == "Distribution of Annual Incomes":
        fig = px.histogram(data, x='annual_inc', nbins=30, title='Distribution of Annual Incomes')
        fig.update_layout(xaxis_title='Annual Income ($)', yaxis_title='Count')
        st.plotly_chart(fig)
    elif dropdown_selection == "Distribution of Interest Rates":
        fig = px.histogram(data, x='int_rate', nbins=30, title='Distribution of Interest Rates')
        fig.update_layout(xaxis_title='Interest Rate (%)', yaxis_title='Count')
        st.plotly_chart(fig)

elif selected_tab == "New Client Default Prediction":
    st.markdown("<h1 class='title-blue'>New Client Default Prediction</h1>", unsafe_allow_html=True)
    st.write("""
    Enter your information to receive a personalized loan recommendation in seconds. Our tool quickly evaluates your eligibility, helping you save time and determine the feasibility of your loan application. If your loan is denied, you will receive a recommendation. If your loan is approved, we will suggest an interest rate.
    """)

    annual_income = st.number_input('Annual Income', value=120000, min_value=0, max_value=1000000)
    loan_term = st.number_input('Loan Term (months)', value=36, min_value=1, max_value=360)
    loan_amount = st.number_input('Loan Amount', value=300000, min_value=0, max_value=1000000)
    home_ownership = st.number_input('Home Ownership (OWN=1, RENT=0)', value=1, min_value=0, max_value=1)
    open_acc = st.number_input('Number of Open Accounts', value=5, min_value=0, max_value=50)
    delinq_2yrs = st.number_input('Delinquencies in Last 2 Years 1=YES 0=NO', value=0, min_value=0, max_value=50)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'annual_inc': [annual_income],
            'term': [loan_term],
            'loan_amnt': [loan_amount],
            'home_ownership_OWN': [home_ownership]
        })

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.markdown(f"<h2 class='title-blue'>Loan Denied</h2>", unsafe_allow_html=True)
            st.write(f"{prediction_proba[0][1] * 100:.2f}% probability of default")
            st.markdown("<h3 class='title-blue'>Recommendations</h3>", unsafe_allow_html=True)
            st.write("""
            - Reduce Loan Amount: A lower loan amount reduces the repayment burden, which can decrease the risk of default.
            - Extend Loan Term: Smaller monthly payments can be easier to manage, reducing the risk of default.
            """)
        else:
            st.markdown(f"<h2 class='title-blue'>Loan Accepted</h2>", unsafe_allow_html=True)
            st.write(f"{prediction_proba[0][1] * 100:.2f}% probability of default")

            # Predict the interest rate using the linear regression model
            input_data_for_rate = pd.DataFrame({
                'loan_amnt': [loan_amount],
                'open_acc': [open_acc],
                'delinq_2yrs': [delinq_2yrs],
                'term': [loan_term]
            })

            predicted_rate = lin_reg_model.predict(input_data_for_rate)
            st.write(f"The suggested interest rate is {predicted_rate[0]:.2f}%.")

elif selected_tab == "Client Risk Segmentation":
    st.markdown("<h1 class='title-blue'>Client Risk Segmentation Analysis</h1>", unsafe_allow_html=True)
    st.write("""
    This heatmap visualizes the risk segmentation of clients based on their loan amounts and annual incomes. Each cell represents the default probability for a specific segment, with colors ranging from green (low risk) to red (high risk). By analyzing this heatmap, we can identify which client segments are more likely to default on their loans, allowing for better risk management and targeted strategies.
    """)

    risk_levels = data.pivot_table(values='loan_status',
                                   index=pd.cut(data['annual_inc'], bins=range(0, 1050000, 100000)),
                                   columns=pd.cut(data['loan_amnt'], bins=range(0, 105000, 10000)),
                                   aggfunc='mean')
    risk_levels = risk_levels.fillna(0)  # fill NaNs with zeros

    x_labels = [f"${i * 100000}" for i in range(11)]  # generate loan amount bins labels
    y_labels = [f"${i * 100000}" for i in range(11)]  # generate annual income bins labels

    heatmap = px.imshow(
        risk_levels.values,
        labels=dict(x="Loan Amount", y="Annual Income", color="Default Probability"),
        x=x_labels[:risk_levels.shape[1]],
        y=y_labels[:risk_levels.shape[0]],
        color_continuous_scale='RdYlGn_r',
    )

    heatmap.update_layout(
        title='Client Risk Segmentation Heatmap',
        xaxis_title='Loan Amount',
        yaxis_title='Annual Income',
        autosize=False,
        width=800,
        height=800
    )

    st.plotly_chart(heatmap)
    st.markdown("<h1 class='title-blue'>Client Risk Evaluation and Interest Rate Recommendations</h1>", unsafe_allow_html=True)
    st.write("""
    We're using our random forest model to calculate a new probability of default for all existing clients. Based on these probabilities, we've also calculated suggested interest rates. The goal is to improve the management of the company's at-risk clients.
    """)

    datatable = data[data['probability_of_default'] < 1].assign(
        client=lambda x: x.index + 1,
        home_ownership=lambda x: x['home_ownership_OWN'].map({1: 'OWN', 0: 'RENT'}),
        suggested_interest_rate=lambda x: lin_reg_model.predict(
            x[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
        ).round(2)
    )

    st.dataframe(datatable[['client', 'annual_inc', 'term', 'loan_amnt', 'home_ownership', 'delinq_2yrs', 'probability_of_default', 'int_rate', 'suggested_interest_rate']])

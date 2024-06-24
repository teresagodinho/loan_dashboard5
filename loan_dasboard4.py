import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('loan_balanced_6040.csv')

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
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Prepare data for linear regression to predict interest rates
X_interest = data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
y_interest = data['int_rate']

# Train a Linear Regression model for predicting interest rates
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_interest, y_interest)

# Standardize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['annual_inc', 'loan_amnt']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Compute default probabilities for each client using the Random Forest Classifier
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:, 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Streamlit app
st.title("Loan Default Prediction Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Tabs", ["Main Page", "Background Information", "New Client Default Prediction", "Client Risk Segmentation"])

if tabs == "Main Page":
    st.write("This dashboard helps a US loan mortgage company identify and manage at-risk clients...")
elif tabs == "Background Information":
    st.write("Explore various graphs that describe our dataset...")

    option = st.selectbox(
        'Select a graph to display',
        ('Correlation Heatmap', 'Distribution of Loan Status', 'Distribution of Loan Amounts', 'Distribution of Annual Incomes', 'Distribution of Interest Rates')
    )

    if option == 'Correlation Heatmap':
        correlation_matrix = data.corr()
        fig = px.imshow(correlation_matrix, 
                        labels={'color':'Correlation'},
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        color_continuous_scale='RdBu_r')
        fig.update_layout(title='Correlation Heatmap')
        st.plotly_chart(fig)

    elif option == 'Distribution of Loan Status':
        loan_status_counts = data['loan_status'].value_counts().reset_index()
        loan_status_counts.columns = ['Loan Status', 'Count']
        fig = px.bar(loan_status_counts, 
                     x='Loan Status', 
                     y='Count',
                     labels={'Loan Status': 'Loan Status', 'Count': 'Number of Loans'},
                     title='Distribution of Loan Status')
        st.plotly_chart(fig)

    elif option == 'Distribution of Loan Amounts':
        fig = px.histogram(data, x='loan_amnt', nbins=50, title='Distribution of Loan Amounts')
        fig.update_layout(xaxis_title='Loan Amount ($)', yaxis_title='Count')
        st.plotly_chart(fig)

    elif option == 'Distribution of Annual Incomes':
        fig = px.histogram(data, x='annual_inc', nbins=50, title='Distribution of Annual Incomes')
        fig.update_layout(xaxis_title='Annual Income ($)', yaxis_title='Count')
        st.plotly_chart(fig)

    elif option == 'Distribution of Interest Rates':
        fig = px.histogram(data, x='int_rate', nbins=50, title='Distribution of Interest Rates')
        fig.update_layout(xaxis_title='Interest Rate (%)', yaxis_title='Count')
        st.plotly_chart(fig)

elif tabs == "New Client Default Prediction":
    st.write("Enter your information to receive a personalized loan recommendation...")

    annual_income = st.number_input('Annual Income', min_value=0, max_value=1000000, value=120000)
    loan_term = st.number_input('Loan Term (months)', min_value=1, max_value=360, value=36)
    loan_amount = st.number_input('Loan Amount', min_value=0, max_value=1000000, value=300000)
    home_ownership = st.selectbox('Home Ownership', ['OWN', 'RENT'])
    home_ownership = 1 if home_ownership == 'OWN' else 0
    open_acc = st.number_input('Number of Open Accounts', min_value=0, max_value=50, value=5)
    delinq_2yrs = st.selectbox('Delinquencies in Last 2 Years', ['YES', 'NO'])
    delinq_2yrs = 1 if delinq_2yrs == 'YES' else 0

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
            result = 'Loan Denied'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"
            st.write(result)
            st.write(probability)
        else:
            result = 'Loan Accepted'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"
            input_data_for_rate = pd.DataFrame({
                'loan_amnt': [loan_amount],
                'open_acc': [open_acc],
                'delinq_2yrs': [delinq_2yrs],
                'term': [loan_term]
            })
            predicted_rate = lin_reg_model.predict(input_data_for_rate)
            recommended_rate = f"The suggested interest rate is {predicted_rate[0]:.2f}%."
            st.write(result)
            st.write(probability)
            st.write(recommended_rate)

elif tabs == "Client Risk Segmentation":
    st.write("Client Risk Segmentation Analysis...")

    risk_levels = data.pivot_table(values='loan_status', 
                                   index=pd.cut(data['loan_amnt'], bins=range(0, 105000, 5000)), 
                                   columns=pd.cut(data['annual_inc'], bins=range(0, 1050000, 50000)), 
                                   aggfunc='mean')
    risk_levels = risk_levels.fillna(0)

    fig = px.imshow(
        risk_levels.values,
        labels=dict(x="Loan Amount", y="Annual Income", color="Default Probability"),
        x=risk_levels.columns.astype(str),
        y=risk_levels.index.astype(str),
        color_continuous_scale='RdYlGn_r',
    )

    fig.update_layout(
        title='Client Risk Segmentation Heatmap',
        xaxis_title='Loan Amount',
        yaxis_title='Annual Income',
        autosize=False,
        width=800,
        height=800
    )

    st.plotly_chart(fig)

    datatable = data[data['probability_of_default'] < 1].assign(
        client=lambda x: x.index + 1,
        home_ownership=lambda x: x['home_ownership_OWN'].map({1: 'OWN', 0: 'RENT'}),
        suggested_interest_rate=lambda x: lin_reg_model.predict(
            x[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
        ).round(2)
    )

    st.dataframe(datatable[['client', 'annual_inc', 'term', 'loan_amnt', 'home_ownership', 'delinq_2yrs', 'probability_of_default', 'suggested_interest_rate']])

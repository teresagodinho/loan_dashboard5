import streamlit as st
import numpy as np
import pandas as pd
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
    'n_estimators': [50, 100],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy']
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

# Standardize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['annual_inc', 'loan_amnt']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Compute default probabilities for each client using the Random Forest Classifier
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:,
                                 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Initialize the Streamlit app
st.set_page_config(layout="wide")
st.title("Loan Default Prediction Dashboard")
st.write(
    """
    This application predicts the probability of loan defaults using a Random Forest classifier. 
    It also provides interest rate recommendations based on a Linear Regression model.
    """
)

# Navigation tabs
tab = st.sidebar.radio("Navigation", ["Main Page", "Background Information", "New Client Default Prediction",
                                      "Client Risk Segmentation"])

if tab == "Main Page":
    st.write(
        """
        This dashboard helps a US loan mortgage company identify and manage at-risk clients. 
        Using machine learning models and statistical analysis, it predicts loan defaults and provides actionable insights. 
        Amid rising US mortgage delinquency rates due to economic uncertainty (Financial Times), this tool enables early 
        identification of potential defaults and better management of at-risk clients, ensuring financial stability and 
        improved loan portfolio management.
        """
    )

elif tab == "Background Information":
    st.write(
        """
        Explore various graphs that describe our dataset, which underpins the predictive tools used in the following tabs. 
        Gain insights into loan distributions, income levels, interest rates, and more.
        """
    )
    dropdown_selection = st.selectbox("Choose a graph to display:",
                                      ["Correlation Heatmap", "Distribution of Loan Status",
                                       "Distribution of Loan Amounts",
                                       "Distribution of Annual Incomes", "Distribution of Interest Rates"])

    if dropdown_selection == 'Correlation Heatmap':
        correlation_matrix = data[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
                                   'delinq_2yrs', 'home_ownership_OWN', 'home_ownership_RENT', 'open_acc',
                                   'loan_status']].corr()
        fig = px.imshow(correlation_matrix,
                        labels={'color': 'Correlation'},
                        x=['Loan Amount', 'Loan Term', 'Interest Rate', 'Installment', 'Annual Income',
                           'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts',
                           'Loan Status'],
                        y=['Loan Amount', 'Loan Term', 'Interest Rate', 'Installment', 'Annual Income',
                           'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts',
                           'Loan Status'],
                        color_continuous_scale='RdBu_r')
        fig.update_layout(title='Correlation Heatmap')
        st.plotly_chart(fig)

    elif dropdown_selection == 'Distribution of Loan Status':
        loan_status_counts = data['loan_status'].value_counts().reset_index()
        loan_status_counts.columns = ['Loan Status', 'Count']
        fig = px.bar(loan_status_counts,
                     x='Loan Status',
                     y='Count',
                     labels={'Loan Status': 'Loan Status', 'Count': 'Number of Loans'},
                     title='Distribution of Loan Status')
        st.plotly_chart(fig)

    elif dropdown_selection == 'Distribution of Loan Amounts':
        fig = px.histogram(data, x='loan_amnt', nbins=50, title='Distribution of Loan Amounts')
        fig.update_layout(xaxis_title='Loan Amount ($)', yaxis_title='Count')
        st.plotly_chart(fig)

    elif dropdown_selection == 'Annual Incomes':
        fig = px.histogram(data, x='annual_inc', nbins=50, title='Distribution of Annual Incomes')
        fig.update_layout(xaxis_title='Annual Income ($)', yaxis_title='Count')
        st.plotly_chart(fig)

    elif dropdown_selection == 'Interest Rates':
        fig = px.histogram(data, x='int_rate', nbins=50, title='Distribution of Interest Rates')
        fig.update_layout(xaxis_title='Interest Rate (%)', yaxis_title='Count')
        st.plotly_chart(fig)

elif tab == "New Client Default Prediction":
    st.write(
        """
        Enter your information to receive a personalized loan recommendation in seconds. Our tool quickly evaluates your 
        eligibility, helping you save time and determine the feasibility of your loan application. If your loan is denied, 
        you will receive a recommendation. If your loan is approved, we will suggest an interest rate.
        """
    )
    annual_income = st.number_input("Annual Income", min_value=0, value=120000)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=300000)
    home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT"])
    home_ownership = 1 if home_ownership == "OWN" else 0
    open_acc = st.number_input("Number of Open Accounts", min_value=0, value=5)
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, value=0)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'annual_inc': [annual_income],
            'term': [loan_term],
            'loan_amnt': [loan_amount],
            'home_ownership_OWN': [home_ownership]
        })

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.write("Loan Denied")
            st.write(f"{prediction_proba[0][1] * 100:.2f}% probability of default")
            st.write(
                """
                - Reduce Loan Amount: A lower loan amount reduces the repayment burden, which can decrease the risk of default.
                - Extend Loan Term: Smaller monthly payments can be easier to manage, reducing the risk of default.
                """
            )
        else:
            st.write("Loan Accepted")
            st.write(f"{prediction_proba[0][1] * 100:.2f}% probability of default")

            input_data_for_rate = pd.DataFrame({
                'loan_amnt': [loan_amount],
                'open_acc': [open_acc],
                'delinq_2yrs': [delinq_2yrs],
                'term': [loan_term]
            })

            predicted_rate = lin_reg_model.predict(input_data_for_rate)
            st.write(f"The suggested interest rate is {predicted_rate[0]:.2f}%.")

elif tab == "Client Risk Segmentation":
    st.write(
        """
        This heatmap visualizes the risk segmentation of clients based on their loan amounts and annual incomes. 
        Each cell represents the default probability for a specific segment, with colors ranging from green (low risk) to red (high risk). 
        By analyzing this heatmap, we can identify which client segments are more likely to default on their loans, allowing for better risk 
        management and targeted strategies.
        """
    )

    risk_levels = data.pivot_table(values='loan_status',
                                   index=pd.cut(data['loan_amnt'], bins=range(0, 105000, 5000)),
                                   columns=pd.cut(data['annual_inc'], bins=range(0, 1050000, 50000)),
                                   aggfunc='mean')
    risk_levels = risk_levels.fillna(0)  # fill NaNs with zeros

    x_labels = [f"${i * 5000}" for i in range(21)]  # generate loan amount bins labels
    y_labels = [f"${i * 50000}" for i in range(21)]  # generate annual income bins labels

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

    st.write(
        """
        We're using our random forest model to calculate a new probability of default for all existing clients. Based on these probabilities, 
        we've also calculated suggested interest rates. The goal is to improve the management of the company's at-risk clients.
        """
    )

    datatable = data[data['probability_of_default'] < 1].assign(
        client=lambda x: x.index + 1,
        home_ownership=lambda x: x['home_ownership_OWN'].map({1: 'OWN', 0: 'RENT'}),
        suggested_interest_rate=lambda x: lin_reg_model.predict(
            x[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
        ).round(2)
    )

    st.dataframe(datatable[['client', 'annual_inc', 'term', 'loan_amnt', 'home_ownership', 'delinq_2yrs',
                            'probability_of_default', 'int_rate', 'suggested_interest_rate']])

if __name__ == '__main__':
    st.run()

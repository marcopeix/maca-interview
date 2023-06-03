import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache_data
def read_sellers_data():
    URL = "https://raw.githubusercontent.com/marcopeix/maca-interview/master/data/sellers_dataset.csv"
    sellers = pd.read_csv(URL, parse_dates=['first_contact_date', 'won_date'])
    return sellers

@st.cache_data
def get_closed_deals(df):
    df['contact_year'] = df['first_contact_date'].dt.year
    df['contact_month'] = df['first_contact_date'].dt.month

    # Calculate the total number of sales per year and month
    total_sales = df.groupby(['contact_year', 'contact_month']).size().reset_index(name='total_sales')

    # Calculate the number of closed sales per year and month
    closed_sales = df.dropna(subset=['seller_id']).groupby(['contact_year', 'contact_month']).size().reset_index(name='closed_sales')

    # Merge total_sales and closed_sales DataFrames
    sales_data = pd.merge(total_sales, closed_sales, on=['contact_year', 'contact_month'], how='left')

    # Calculate the percentage of closed sales per year and month
    sales_data['percentage_closed_sales'] = (sales_data['closed_sales'] / sales_data['total_sales']) * 100

    sales_data = sales_data.fillna(value=0)

    return sales_data

@st.cache_data
def get_closed_deals_per_origin(df):
    # Calculate the total number of deals per origin
    total_deals = df.groupby('origin').size().reset_index(name='total_deals')

    # Calculate the number of closed deals per origin
    closed_deals = df.dropna(subset=['seller_id']).groupby('origin').size().reset_index(name='closed_deals')

    # Merge total_deals and closed_deals DataFrames
    deal_data = pd.merge(total_deals, closed_deals, on='origin', how='left')

    # Calculate the percentages of closed deals per origin
    deal_data['percentage_closed'] = (deal_data['closed_deals'] / deal_data['total_deals']) * 100

    deal_data = deal_data.sort_values(by='percentage_closed', ascending=False)

    return deal_data

@st.cache_data
def compute_average_close_time(df):
    df = df.dropna(subset=['seller_id'], axis=0)

    # Calculate the time difference in days
    df['time_difference'] = (df['won_date'] - df['first_contact_date']).dt.days

    # Calculate the average time difference in days
    avg_close_time = df['time_difference'].mean()

    return round(avg_close_time,0)

@st.cache_data
def compute_avg_declared_monthly_revenue(df):
    df = df.dropna(subset=['seller_id'], axis=0)

    avg_monthly_rev = df['declared_monthly_revenue'].mean()

    return round(avg_monthly_rev, 2)

@st.cache_data
def compute_shap(df):
    model_df = df.dropna(subset=['seller_id'], axis=0)
    model_df = model_df[['origin', 'business_segment', 'business_type', 'declared_monthly_revenue']]
    model_df = model_df.dropna(axis=0)
    model_df_encoded = pd.get_dummies(model_df.drop('declared_monthly_revenue', axis=1))

    X_train, X_test, y_train, y_test = train_test_split(model_df_encoded, model_df['declared_monthly_revenue'], test_size=0.2, random_state=42)
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    
    explainer = shap.Explainer(dt)
    shap_values = explainer(X_test)

    vals= np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['factor','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    
    return feature_importance['factor'].to_frame()

@st.cache_data
def forecast_sales(df):
    endog = df['closed_sales']

    best_lags = ar_select_order(endog, maxlag=5).ar_lags

    p = best_lags[-1]

    model = SARIMAX(endog, order=(p,0,0))
    res = model.fit(disp=False)
    preds = res.get_forecast(6)

    conf_int_df = preds.conf_int()
    preds_df = conf_int_df.merge(preds.predicted_mean.to_frame(), left_index=True, right_index=True)

    preds_df = preds_df.applymap(round)
    preds_df = preds_df.applymap(lambda x: int(x))
    preds_df = preds_df.applymap(lambda x: 0 if x < 0 else x)

    return preds_df


def display_leads_conversion(df):
    fig, ax = plt.subplots()
    ax.plot(pd.date_range('2017-06-01', '2018-05-31', freq='M').floor('D'), df['percentage_closed_sales'])
    ax.set_ylabel('Percentage of closed sales')
    ax.set_xlabel('Date')
    fig.autofmt_xdate()
    st.pyplot(fig)

def display_deals_origin(df):
    x = df['origin']
    y = df['percentage_closed']

    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_ylabel('Percentage of closed sales')
    ax.set_xlabel('Origin of contact')
    ax.set_ylim(0,20)
    for i, v in enumerate(y):
        ax.text(x=i, y=v+0.5, s=str(round(v,2)), ha='center')
    fig.autofmt_xdate()
    st.pyplot(fig)

def display_forecast(df, preds_df):
    fig, ax = plt.subplots()

    ax.plot(df['closed_sales'])
    ax.plot(preds_df['predicted_mean'])
    ax.fill_between(preds_df.index, preds_df['lower closed_sales'], preds_df['upper closed_sales'], alpha=0.1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Monthly sales')
    ax.set_xticks(np.arange(0,18,2), pd.date_range(start=start_date, periods=9, freq='M').date)
    fig.autofmt_xdate()
    st.pyplot(fig)

if __name__ == "__main__":
    st.title('RevOPS Dashboard')

    with st.form("date_form"):
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Pick a start date", 
                value=datetime.date(2017,6,14), 
                min_value=datetime.date(2017,6,14), 
                max_value=datetime.date(2018,5,31))
        with col2:
            end_date = st.date_input(
                "Pick an end date", 
                value=datetime.date(2018,5,31), 
                min_value=datetime.date(2017,6,14), 
                max_value=datetime.date(2018,5,31))

        st.write("Filtering with the date might not work, maybe I won't have time to implement this.")
        st.form_submit_button("Analyze", type="primary")

    sellers = read_sellers_data()
    closed_deals_df = get_closed_deals(sellers)
    closed_deal_origin = get_closed_deals_per_origin(sellers)
    avg_close_time = compute_average_close_time(sellers)
    avg_monthly_rev = compute_avg_declared_monthly_revenue(sellers)

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Average close time', value=f"{avg_close_time} days")
    with col2:
        st.metric("Average seller's monthly revenue", value=f"${avg_monthly_rev}")

    st.subheader('Percentage of deals closed')
    display_leads_conversion(closed_deals_df)

    st.subheader('Percentage of closed deals per origin')
    display_deals_origin(closed_deal_origin)

    st.header("Insights")

    st.subheader("Factors likely to increase revenue")
    shap_df = compute_shap(sellers)
    st.write(shap_df.head())

    st.subheader("Sales forecast")
    preds_df = forecast_sales(closed_deals_df)
    display_forecast(closed_deals_df, preds_df)
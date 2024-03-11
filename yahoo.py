import streamlit as st
import pandas as pd
import numpy as np
# Get time series data
import yfinance as yf

# Prophet model for time series forecast
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Data processing
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
# import matplotlib.pyplot as plt

# Model performance evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid



# Add CSS for background image
st.set_page_config(page_title="Yahoo finance prediction", page_icon="📈")
# Title of the app

st.title('Yahoo finance stock prediction')

# Use columns to create a sidebar-like layout
col1, col2 = st.columns([1, 3])
def exp_goog():
            exp=data.copy()
            train_d = exp[exp['ds'] <= train_end_date]
            test_d = exp[exp['ds'] > train_end_date]

            train=train_d['y'].values
            test=test_d['y'].values
            # Step 3: Splitting Data

            # Model Selection and Training
            model_es = ExponentialSmoothing(train)
            model_es_fit = model_es.fit()

            # Forecasting
            forecast_es = model_es_fit.forecast(len(test))
            # Calculate RMSE
            exp_rmse = np.sqrt(mean_squared_error(test, forecast_es))
            exp_mape = np.mean(np.abs(test - forecast_es) / test)

            # Print RMSE and MAPE
            st.write("Exponential Smoothing RMSE on test data:", exp_rmse)
            st.write("Exponential Smoothing MAPE on test data:", exp_mape)
def prophet_uni():
        ## PROPHERT UNIVARIATE - GOOG

    prophet_uni_data=data[['ds','y']].copy()
    prophet_uni_data.reset_index(inplace=True)

    train = prophet_uni_data[prophet_uni_data['ds'] <= train_end_date]
    test = prophet_uni_data[prophet_uni_data['ds'] > train_end_date]

    try:
        prophet = Prophet(growth="linear",
                        changepoint_range=0.8, changepoint_prior_scale=0.5,
                        yearly_seasonality=True, weekly_seasonality=False)
        prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        prophet.fit(train)
        #print("Prophet model fitting completed successfully.")
    except Exception as e:
        st.write("Error occurred during Prophet model fitting:", str(e))

    # Make predictions on test set
    forecast_prophet_uni = prophet.predict(test)

    # Calculate RMSE
    prophet_uni_rmse = np.sqrt(mean_squared_error(test['y'], forecast_prophet_uni['yhat']))

    # Calculate MAPE
    prophet_uni_mape = mean_absolute_percentage_error(test['y'], forecast_prophet_uni['yhat'])

    st.write("RMSE:", prophet_uni_rmse)
    st.write("MAPE:", prophet_uni_mape)
def exp_tsla():
    #ExponentialSmoothing

    exp=data.copy()
    exp=exp[['ds','TSLA']]
    exp=exp.rename(columns={'TSLA':'y'})
    train_d = exp[exp['ds'] <= train_end_date]
    test_d = exp[exp['ds'] > train_end_date]

    train=train_d['y'].values
    test=test_d['y'].values
    # Step 3: Splitting Data

    # Model Selection and Training
    model_es = ExponentialSmoothing(train)
    model_es_tsla_fit = model_es.fit()

    # Forecasting
    forecast_tsla_es = model_es_tsla_fit.forecast(len(test))
    train_end_date = pd.to_datetime(train_end_date)
    future_dates = pd.date_range(start=train_end_date + pd.DateOffset(days=1), periods=24, freq='M')

    # Calculate RMSE
    exp_tsla_rmse = np.sqrt(mean_squared_error(test, forecast_tsla_es))
    exp_tsla_mape = np.mean(np.abs(test - forecast_tsla_es) / test)

    forecast_es_future = model_es_tsla_fit.forecast(len(future_dates))

    future_df = pd.DataFrame({
        'ds': future_dates,
        'TSLA': forecast_es_future,
        'test': [test[0] if i < 12 else np.nan for i in range(24)]
    })

    # Display the future DataFrame

    all_months=(future_df[pd.isna(future_df['test'])])
    tsla_pred=all_months[['ds','TSLA']]
    return tsla_pred

def prophet_uni_hp():
    
    ## PROPHET UNI HYPER PARAMETER TUNED
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'changepoint_range': [0.8, 0.9],
        'changepoint_prior_scale': [0.1, 0.5, 1.0],
    }

    # Define variables
    prophet_uni = data[['ds', 'y']].copy()
    train = prophet_uni[prophet_uni['ds'] <= train_end_date]
    test = prophet_uni[prophet_uni['ds'] > train_end_date]

    prophet_uni_hp_params = None
    prophet_uni_hp_rmse = float('inf')
    prophet_uni_hp_mape = float('inf')

    for params in ParameterGrid(param_grid):
        try:
            # Initialize and fit Prophet model
            prophet = Prophet(growth="linear",
                            changepoint_range=params['changepoint_range'],
                            changepoint_prior_scale=params['changepoint_prior_scale'],
                            yearly_seasonality=False, weekly_seasonality=False)
            prophet.fit(train)

            # Make predictions on test set
            forecast_prophet_uni = prophet.predict(test)

            # Calculate RMSE and MAPE
            prophet_uni_rmse = np.sqrt(mean_squared_error(test['y'], forecast_prophet_uni['yhat']))
            prophet_uni_mape = mean_absolute_percentage_error(test['y'], forecast_prophet_uni['yhat'])

            # Update best parameters if current model is better
            if prophet_uni_rmse < prophet_uni_hp_rmse:
                prophet_uni_hp_rmse = prophet_uni_rmse
                prophet_uni_hp_mape = prophet_uni_mape
                prophet_uni_hp_params = params

        except Exception as e:
            print("Error occurred during Prophet model fitting:", str(e))
            continue
    
    st.write("Best parameters:", prophet_uni_hp_params)
    st.write("Best RMSE:", prophet_uni_hp_rmse)
    st.write("Best MAPE:", prophet_uni_hp_mape)
def prophet_multi():
    # PROPHET MULTI NORMAL
    prophet_multi=data[['ds','y','TSLA']].copy()
    # Split prophet_multi into train and test sets
    train = prophet_multi[prophet_multi['ds'] <= train_end_date]
    test = prophet_multi[prophet_multi['ds'] > train_end_date]

    try:
        prophet = Prophet(growth="linear",
                        changepoint_range=0.8, changepoint_prior_scale=0.5,
                        yearly_seasonality=False, weekly_seasonality=False)

        # Add additional regressors
        prophet.add_regressor('TSLA')

        prophet.fit(train)
        print("Prophet model fitting completed successfully.")
    except Exception as e:
        st.write("Error occurred during Prophet model fitting:", str(e))

    # Make predictions on test set
    forecast_prophet_multi = prophet.predict(test)

    # Calculate RMSE
    prophet_multi_rmse = np.sqrt(mean_squared_error(test['y'], forecast_prophet_multi['yhat']))

    # Calculate MAPE
    prophet_multi_mape = mean_absolute_percentage_error(test['y'], forecast_prophet_multi['yhat'])

    st.write("RMSE:", prophet_multi_rmse)
    st.write("MAPE:", prophet_multi_mape)


def prophet_multi_hp():
    prophet_multi=data[['ds','y','TSLA']].copy()
# Split prophet_multi into train and test sets
    train = prophet_multi[prophet_multi['ds'] <= train_end_date]
    test = prophet_multi[prophet_multi['ds'] > train_end_date]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.2,0.4,0.3],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0,0.2,0.3],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0,0.3,0.5],
        # Add more hyperparameters to tune
    }

    # Initialize variables to store the best model and its metrics
    prophet_multi_hp_best_params = None
    prophet_multi_hp_rmse = np.inf
    prophet_multi_hp_mape = np.inf

    # Iterate over each combination of hyperparameters
    for params in ParameterGrid(param_grid):
        try:
            # Initialize Prophet model with the current set of hyperparameters
            prophet = Prophet(growth="linear", yearly_seasonality=True, interval_width=0.95, weekly_seasonality=False, **params)

            # Add additional regressors
            prophet.add_regressor('TSLA')

            # Fit the model on the training data
            prophet.fit(train)

            # Make predictions on the test set
            forecast = prophet.predict(test)

            # Calculate RMSE and MAPE
            rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
            mape = mean_absolute_percentage_error(test['y'], forecast['yhat'])

            # Check if the current model is better than the previous best model
            if rmse < prophet_multi_hp_rmse:
                prophet_multi_hp_rmse = rmse
                prophet_multi_hp_mape = mape
                prophet_multi_hp_best_params = prophet

        except Exception as e:
            print("Error occurred during Prophet model fitting:", str(e))

    
# Print the best model's metrics
    st.write("Prophet multivariate hypertuned RMSE:", prophet_multi_hp_rmse)
    st.write("Prophet multivariate hypertuned MAPE:", prophet_multi_hp_mape)



def arima():
    
        # Splitting the data into train and test sets based on the dates
        train = data[data['ds'] <= train_end_date]['y'].values
        test = data[data['ds'] > train_end_date]['y'].values

        param_grid = {
            'p': range(0, 6),  # AR parameter range
            'd': range(0, 2),  # Integration parameter range
            'q': range(0, 6)   # MA parameter range
        }

        arima_mape = np.inf
        arima_best_params = None

        for params in ParameterGrid(param_grid):
            try:
                # Fit ARIMA model
                model_arima = ARIMA(train, order=(params['p'], params['d'], params['q']))
                model_arima_fit = model_arima.fit()

                # Forecast ARIMA model on test set
                forecast_arima = model_arima_fit.forecast(steps=len(test))

                # Calculate MAPE
                mape = np.mean(np.abs((test - forecast_arima) / test)) * 100

                # Update best parameters if a better one is found
                if mape < arima_mape:
                    arima_mape = mape
                    arima_best_params = params
                    best_forecast = forecast_arima

            except:
                continue

        # Train the final model with the best parameters
        final_model = ARIMA(train, order=(arima_best_params['p'], arima_best_params['d'], arima_best_params['q']))
        final_model_fit = final_model.fit()

        # Forecast ARIMA model on test set
        forecast_arima = final_model_fit.forecast(steps=len(test))

        # Calculate RMSE on test set
        arima_rmse = np.sqrt(mean_squared_error(test, forecast_arima))
        st.write("RMSE on test data:", arima_rmse)
        st.write("MAPE on test data:", arima_mape)
# Add navigation options in the first column
with col1:
    st.write("**Navigation**")
    page = st.radio("Go to", ["Yahoo finance basics", "Time series model basics", "Explaining data","Output"])

# Partition 1: Section 1
with col2:
    if page == "Yahoo finance basics":
        st.header('Yahoo finance basics')
        st.markdown(""" 
            **Opening price**: The opening price reflects the initial price at which the Company's stock started trading each day. For example, on Monday, the opening price was $100, indicating the price at which the first trade occurred when the market opened. The opening price provides insights into the market sentiment at the beginning of the trading session.

            **Closing price**: The closing price represents the final price at which the stock is traded at the end of each trading day. For instance, if the closing price was $104, indicating the last trade executed before the market closed. The closing price is significant because it reflects the sentiment and activity of investors throughout the trading day.

            **Adjusted Closing Price**: The adjusted closing price is the closing price adjusted for corporate actions such as dividends, stock splits, and mergers. In our example, the adjusted closing price accounts for any changes in the stock's price due to these events. For instance, if there was a dividend payout, the adjusted closing price would reflect this adjustment to provide a more accurate representation of the stock's performance.

            **High Price**: The high price, also known as the high, represents the highest price at which a particular asset, such as a stock, commodity, or currency pair, was traded during a specific time frame.

            **Low Price**: Conversely, the low price, or low, denotes the lowest price at which the asset traded during the same time frame. It represents the lowest level reached by the asset's price within the specified period.

            **Volume**: Volume refers to the total number of shares traded during a specific period, such as a trading day. It indicates the level of activity and interest in the stock.
        """)
    elif page == "Time series model basics":
        st.header('Time series model basics - Introduction')
        st.markdown(""" 
            **Why Time Series Models?**: Time series models are essential for analyzing data that changes over time, such as stock prices, weather patterns, and economic indicators. These models help us understand past trends, forecast future values, and make informed decisions.

            **Difference from Machine Learning Models**: While machine learning models focus on generalizing patterns and making predictions based on features, time series models specifically handle temporal dependencies and capture seasonality, trends, and periodic patterns inherent in time series data.

            **Types of Time Series Models**:
            - **ARIMA (AutoRegressive Integrated Moving Average)**: A classical approach that models the next step in the sequence as a linear function of the observations at prior time steps.
            - **Exponential Smoothing**: A family of models that use exponential weighting to capture short-term and long-term patterns in data.
            - **Prophet (Univariate and Multivariate)**: Developed by Facebook, Prophet is designed for forecasting time series data that display patterns on different time scales, such as yearly, weekly, and daily patterns.
        """)
    elif page == "Explaining data":
        st.header('Explaining Data')
        st.markdown(""" 
                Pull stock data from Yahoo Finance API. We will pull 10 years of daily data from the beginning of 2014 to the Feb 2024.(Yahoo finance [Link](https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAABC-GJ2RNJi9guUKvWj4Lg23PWRwqmk1SRlEEM7fAPcTbzP-ljhSobnY9ExfQN2_Ept7N2kgCjXi3Xrm2Dnw_cfbaJ7YnLNBYK_vGcjzpxW-nHz_QIVNtnBjuJe5Y7QiBm3ZLDzMCIRnGMXib4-gKoaQrFw8wWNQDNJf-kJpPJwl))
                    
                **INPUTS:** I downloaded the closing prices for two tickers, **GOOG** and **TSLA**. GOOG is the ticker for Google, and TSLA is the ticker for the TSLA Equity - NMS. 
                    

                **GOAL:** The goal of the time series model is to predict the closing price of Google stock. The closing price of TSLA will be used as an additional predictor.

                **TWEAKS MADE :** 
                - The data is present in day level having 2557 entries. The main aim is to predict close value for 2023. So, aggregated the data on month level . Now the data contains 122 rows. 
                - For testing data, I have considered 2023 and for prediction, I'll be predicting next 12 months GOOG Close value
                - For multivariate model, we need the data for next 12 months. So, instead of taking it from internet, I used a univariate model to predict the variable which will be fed as input while predicting the multivariate model 
                - The prediction is based on the fact that previous pattern behaviour will continue in the future
                - For Multivariate, I'm considering TSLA as the variable since it is having a good correlation of 0.93 (Strong positive correlation) with GOOG. 
                    
                **DATE RANGE :**  
                - **start_date** = 2014-01-02 because January 1st is a holiday, and there is no stock data on holidays and weekends.                    
                - **end_date** = 2024-02-01 because yfinance excludes the end date, so we need to add one day to the last day of the data end date.
                - **train_end_date** = 2023-02-01 
                - Predict next 12 months of data using the best model
                    
                **FINAL PREDICTOR :** Here I'm considering the column **Close** as the target variable and we will be predicting the Close value for next 12 months
                    
                **METRICS USED :** 2 main metrics are used here : 
                - **MAPE**
                - **RMSE** """)
        start_date = '2014-01-02'

        # Data end date. yfinance excludes the end date, so we need to add one day to the last day of data
        end_date = '2024-03-01' 

        # Date for splitting training and testing dataset
        train_end_date = '2023-03-01'
        # Pull close data from Yahoo Finance for the list of tickers
        ticker_list = ['GOOG', 'TSLA']
        data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

        
        data.index = pd.to_datetime(data.index)
        # Resample and aggregate close prices by month
        data = data.resample('M').mean()
        
        # Change variable names
        data = data.reset_index()
        st.write("Input table looks like this: ")
        st.table(data.head())

        
    elif page=='Output':
        st.write("Google Colab notebook [Link](https://colab.research.google.com/drive/1Is6s1-qpP354ys9yR6biTN_NzAsCNlgI?usp=sharing)")
        
        st.header('OUTPUT OF VARIOUS MODELS USED TO PREDICT THE STOCK CLOSING PRICE OF GOOG')
        # Data start date
        start_date = '2014-01-02'

        # Data end date. yfinance excludes the end date, so we need to add one day to the last day of data
        end_date = '2024-03-01' 

        # Date for splitting training and testing dataset
        train_end_date = '2023-03-01'
        # Pull close data from Yahoo Finance for the list of tickers
        ticker_list = ['GOOG', 'TSLA']
        data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

        
        data.index = pd.to_datetime(data.index)
        # Resample and aggregate close prices by month
        data = data.resample('M').mean()
        
        # Change variable names
        data = data.reset_index()
        #st.write("Input table looks like this: ")
        #st.table(data.head())
        data.columns = ['ds', 'y', 'TSLA']
        
        #ExponentialSmoothing
        option = st.selectbox(
        'You want to see output of which model?',
        ('ARIMA', 'Exponential smoothing', 'PROPHET Univariate','PROPHET Univariate Hyper tuned','PROPHET Multivariate','PROPHET Multivariate Hypertuned','ALL'))
        st.write('You selected:', option)
        if option=="ARIMA":
            st.header("ARIMA OUTPUT METRICS")
            arima()
        elif option=="Exponential smoothing":
            st.header("EXPONENTIAL SMOOTHING OUTPUT METRICS")
            exp_goog()   
        elif option == "PROPHET Univariate" :
            st.header("PROPHET UNIVARIATE METRICS OUTPUT")
            prophet_uni()    
        elif option=="PROPHET Univariate Hyper tuned":
            st.header("PROPHET UNIVARIATE (WITH HYPER TUNED PARAMETERS) METRICS OUTPUT")
            prophet_uni_hp()
        elif option=="PROPHET Multivariate":
            st.header("PROPHET Multivariate METRICS OUTPUT")
            prophet_multi()
        elif option=="PROPHET Multivariate Hypertuned":
            st.header("PROPHET Multivariate with hyper tuned paramenters METRICS OUTPUT")
            prophet_multi_hp()
        
        else:
            data = {
            'Model': ['ARIMA','PROPHET UNIVARIATE (NORMAL)','PROPHET MULTIVARIATE (NORMAL)','PROPHET UNIVARIATE (HYPERPARAMETER TUNED)','PROPHET MULTIVARIATE (HYPER PARAMETER TUNED)','EXPONENTIAL SMOOTHING'],
            'RMSE': [19.85,31.67,34.43,13.33,10.23,33.78],
            'MAPE (%)': [13.65,21.28,22.9,8.64,6.73,23.1]}
            df = pd.DataFrame(data)
               # Display the DataFrame as a table
            st.table(df)
            st.markdown ("""
                            Prophet multivariate with hypertuned parameters has the least MAPE and RMSE
                            The prediction for next 12 months using prophet is """)
            data = {
                    'Date': ['31-3-2024','30-4-2024','31-5-2024','30-6-2024','31-7-2024','31-8-2024','30-9-2024','31-10-2024','30-11-2024','31-12-2024','31-1-2025','28-2-2025'],
                    'Close value prediction': [134.76,135.19,135.87,137.07,138.90,140.42,139.90,140.31,141.76,142.34,143.36,144.94]
            df = pd.DataFrame(data)
                # Display the DataFrame as a table
            st.table(df)
                    
                                
            
    # Assuming you have defined 'data', 'train_end_date', and 'test_end_date' somewhere before




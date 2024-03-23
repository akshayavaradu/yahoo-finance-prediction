
from fastapi import FastAPI
from pydantic import BaseModel

import streamlit as st
import pandas as pd
import numpy as np
# Get time series data
import yfinance as yf
import json
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



# def kamal1():
#     print("My Streamlit App")
#     print("Hello, world!")

# class Outputs(BaseModel):
#     item1: float
#     item2: float
#     item3: list

app = FastAPI()


@app.get("/arima/")
async def arima():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']

    # Take a look at the data
    ## to display data in a table format
    # st.table(data.head())
    
    
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
    arima_rmse = np.mean(np.abs(test - forecast_arima) / test)
    return {
        "Arima_rmse": arima_rmse,
        "Arima_mape": arima_rmse,
    }
    
    
@app.get("/exponential_smoothing/")
async def exponential_smoothing():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']

    # Take a look at the data
    ## to display data in a table format
    # st.table(data.head())
    
    
    #ExponentialSmoothing
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
    
    return {
        "exponential_smoothing_rmse": exp_rmse,
        "exponential_smoothing_mape": exp_mape,
    }
    
    # uvicorn yfinance_api:app --reload
@app.get("/prophet_uni/")
async def prophet_uni():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']

    # Take a look at the data
    ## to display data in a table format
    # st.table(data.head())
    
    
    #ExponentialSmoothing
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

    return {
        "prophet_uni_rmse": prophet_uni_rmse,
        "prophet_uni_mape": prophet_uni_mape,
    }
    
    # uvicorn yfinance_api:app --reload
@app.get("/prophet_multi/")
async def prophet_multi():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']

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

    return {
        "prophet_multi_rmse": prophet_multi_rmse,
        "prophet_multi_mape": prophet_multi_mape,
    }
    
    # uvicorn yfinance_api:app --reload
@app.get("/prophet_uni_hyperparameters/")
async def prophet_uni_hyperparameters():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']
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
    


    return {
        "prophet_univariate_with_hyperparameters_rmse": prophet_uni_hp_rmse,
        "prophet_univariate_with_hyperparameters_mape": prophet_uni_hp_mape,
    }
    
    # uvicorn yfinance_api:app --reload
@app.get("/prophet_multi_hyperparameters/")
async def prophet_multi_hyperparameters():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']
 

    # Define variables
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
    return {
        "prophet_univariate_with_hyperparameters_rmse": prophet_multi_hp_rmse,
        "prophet_univariate_with_hyperparameters_mape": prophet_multi_hp_mape,
    }
    

    # uvicorn yfinance_api:app --reload
@app.get("/final_prediction/")
async def final_prediction():
    
    start_date = '2014-01-02'
    end_date = '2024-03-01' 

    train_end_date = '2023-03-01'

    ticker_list = ['GOOG', 'TSLA']
    data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

    data.index = pd.to_datetime(data.index)

    data = data.resample('M').mean()
    # Change variable names
    data = data.reset_index()
    data.columns = ['ds', 'y', 'TSLA']
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
   

    # Define variables
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
    forecast_final=prophet_multi_hp_best_params.predict(tsla_pred)
    forecast_final['ds'] = forecast_final['ds'].astype(str)

    # Convert the DataFrame to a dictionary
    forecast_dict = forecast_final[['ds', 'yhat']].to_dict(orient='records')

    # Serialize the dictionary to JSON
    forecast_json = json.dumps(forecast_dict)
    forecast_data = json.loads(forecast_json)

    # Construct the new dictionary with the desired format
    new_forecast_dict = {}
    for item in forecast_data:
        new_forecast_dict[item['ds']] = round(item['yhat'], 2)

# Convert to JSON format
    output_json = '{'
    for key, value in new_forecast_dict.items():
        output_json += f'"{key}": {value}, '
    output_json = output_json[:-2]  # Remove the last comma and space
    output_json += '}'

    output_json_1 = output_json.replace("\\", "")

# Print the JSON string without backslashes
    
    # Return the JSON string
    return output_json_1
    # uvicorn yfinance_api:app --reload
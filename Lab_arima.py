import math
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

class Arima_Lab:

    @staticmethod
    def query_range_date(city, start, end):
        return {"city": 'Milano',
         "init_time": {"$gte": start, "$lte": end}}
    @staticmethod
    def query_outliers(thresh_down, thresh_up):
        return {"$expr": {"$ne": ["$init_address", "$final_address"]},
                      "duration": {"$gte": thresh_down, "$lte": thresh_up}}

    @staticmethod
    def mean_variance(db, start, end, city):

        query_date = Arima_Lab.query_range_date(city, start, end)
        pipeline = [
            {"$match": query_date},
            {"$project": {
                "city": 1,
                "init_time": 1,
                "final_time": 1,
                "duration": {"$subtract": ["$final_time", "$init_time"]}
            }
            },
            {"$group": {
                "_id": None,
                "mean": {"$avg": "$duration"},
                "variance": {"$stdDevPop": "$duration"}
            }
            }
        ]

        return pd.DataFrame(list(db.aggregate(pipeline)))
    @staticmethod
    def remove_outliers(db, start, end, city, thresh_down, thresh_up):
        projection = {
            "_id": 0,
            "city": 1,
            "init_time": 1,
            "init_date": 1,
            "init_address": 1,
            "final_address": 1,
            "duration": {"$subtract": ["$final_time", "$init_time"]},
            "year": {"$year": "$init_date"},
            "month": {"$month": "$init_date"},
            "week": {"$week": "$init_date"},
            "day": {"$dayOfMonth": "$init_date"},
            "hour": {"$hour": "$init_date"}
        }

        pipeline = [
            {
                "$match": Arima_Lab.query_range_date(city, start, end)
            },
            {
                "$project": projection
            },
            {
                "$match": Arima_Lab.query_outliers(thresh_down, thresh_up)
            },
            {
                "$group": {
                    "_id": {
                        "year": "$year",
                        "month": "$month",
                        "week": "$week",
                        "day": "$day",
                        "hour": "$hour"
                    },
                    "number": {"$sum": 1}}},
            {"$project": {
                "_id": 0,
                "number": 1,
                "year": "$_id.year",
                "month": "$_id.month",
                "week": "$_id.week",
                "day": "$_id.day",
                "hour": "$_id.hour"
            }
            }
        ]

        return pd.DataFrame(list(db.aggregate(pipeline)))

    @staticmethod
    def processing(outliers_removed_df):
        outliers_removed_df['date'] = pd.to_datetime(outliers_removed_df[["year", "month", "day", "hour"]])
        outliers_removed_df = outliers_removed_df.sort_values(by="date")
        # print(outliers_removed_df.head())

        date_range = pd.date_range(start=outliers_removed_df['date'].min(), end=outliers_removed_df['date'].max(),
                                   freq='H')
        all_dates_df = pd.DataFrame({'date': date_range})
        merged_df = pd.merge(all_dates_df, outliers_removed_df, on='date', how='left')

        mean_values = outliers_removed_df.groupby(['day', 'hour'])['number'].mean().reset_index()

        merged_df['number'].fillna(merged_df.groupby(['week', 'day', 'hour'])['number'].transform('mean'), inplace=True)

        mean_value = merged_df['number'].mean()
        merged_df['number'].fillna(mean_value, inplace=True)

        #print(merged_df)

        return merged_df

    @staticmethod
    def plot_n_cars(merged_df, city):
        plt.figure(figsize=(10, 6))
        plt.plot(merged_df['date'], merged_df['number'])
        plt.title(f"Number of cars in {city}")
        plt.xlabel('Date')
        plt.ylabel('Number of cars')
        plt.xticks(rotation=45)
        plt.savefig(f"n_of_cars_{city}.png")# Rotate x-axis labels for better readability
        plt.show()

        plt.figure()
        pd.plotting.autocorrelation_plot(merged_df['number'])
        plt.title(f"Autocorrelation with pandas for city of {city}")
        plt.savefig(f"first_autocorr_{city}.png")
        plt.show()

    @staticmethod
    def diff_plot(merged_df, city):
        diff_data = merged_df.copy()
        diff_data['first_diff'] = diff_data['number'].diff()
        diff_data['second_diff'] = diff_data['first_diff'].diff()

        # print(diff_data['number'])

        plt.figure()
        plt.plot(diff_data['date'], diff_data['first_diff'], label='First diff')
        plt.plot(diff_data['date'], diff_data['second_diff'], label='Second diff')
        plt.xlabel('Date')
        plt.ylabel('Number of cars')
        plt.xticks(rotation=45)
        plt.title(f"Differentiated data for city of {city}")
        plt.legend()
        plt.savefig(f"diff_data_{city}.png")
        plt.show()
        return diff_data

    @staticmethod
    def autocorr(diff_data, n_lags, city):
        acf = sm.tsa.acf(diff_data['number'], nlags=n_lags)
        plt.subplot()
        plt.plot(acf)
        plt.axis([0, 24, -.5, 1])
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(diff_data)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(diff_data)), linestyle='--', color='gray')
        plt.title(f"Autocorrelation Function for city of {city}")
        plt.savefig(f"autocorr_{city}.png")
        plt.show()
        return acf

    @staticmethod
    def partial_autocorr(diff_data, n_lags, city):
        p_acf = sm.tsa.pacf(diff_data['number'], nlags=n_lags)
        plt.subplot()
        plt.plot(p_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(diff_data)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(diff_data)), linestyle='--', color='gray')
        plt.title(f"Partial Autocorrelation Function for city of {city}")
        plt.savefig(f"p_autocorr_{city}.png")
        plt.show()
        return p_acf

    @staticmethod
    def filter(values, threshold):
        list_of_pos = []
        for i in range(len(values)):
            if values[i] > threshold:
                list_of_pos.append(i)

        return np.array(list_of_pos)

    @staticmethod
    def train_test(diff_data, samples_train, samples_test):
        diff_data = diff_data.set_index('date')

        first_day = datetime(2017, 10, 1)

        train_end = first_day + pd.to_timedelta(samples_train, unit='h')
        test_end = train_end + pd.to_timedelta(samples_test, unit='h')
        train = diff_data.loc[first_day:train_end, ['number']]
        test = diff_data.loc[train_end:test_end, ['number']]
        return train, test

    @staticmethod
    def simple_ARIMA(train, test, p, d, q, city):
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        print(model_fit.summary())
        forecast = model_fit.get_forecast(steps=len(test))
        plt.plot(forecast.predicted_mean.values)
        plt.title(f"ARIMA MODEL in {city}, p={p}, d={d}, q={q}")
        plt.plot(np.array(test))
        plt.savefig(f"simple_ARIMA_{city}.png")
        plt.show()
        print(f'MAE: %.3f', mean_absolute_error(test.values, forecast.predicted_mean.values))
        print(f'MAPE: %.3f', (mean_absolute_error(test.values, forecast.predicted_mean.values) / test['number'].mean()) * 100)
        print(f'MSE %.3f', mean_squared_error(test.values, forecast.predicted_mean.values))
        print(f'R2 %.3f', r2_score(test.values, forecast.predicted_mean.values))

    @staticmethod
    def grid_search(train, test, acf_values, p_acf_values, city):
        model = ARIMA(train, order=(0, 0, 0))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=len(test))
        mae = mean_absolute_error(test.values, forecast.predicted_mean.values)
        mape = mae / test['number'].mean() * 100
        mse = mean_squared_error(test.values, forecast.predicted_mean.values)
        r2 = r2_score(test.values, forecast.predicted_mean.values)
        df = pd.DataFrame({'d': [0], 'p': [0], 'q': [0], 'MAE': [mae], 'MAPE': [mape], 'mse': [mse], 'r2_score': [r2]})

        d_index = [0, 1, 2]

        for d in d_index:
            for q in acf_values:
                for p in p_acf_values:
                    model = ARIMA(train, order=(q, d, p))
                    print('(d: ', d, 'q: ', q, 'p: ', p, ')')
                    model_fit = model.fit()
                    forecast = model_fit.get_forecast(steps=len(test))
                    mae = mean_absolute_error(test.values, forecast.predicted_mean.values)
                    mape = mae / test['number'].mean() * 100
                    mse = mean_squared_error(test.values, forecast.predicted_mean.values)
                    r2 = r2_score(test.values, forecast.predicted_mean.values)
                    new_row = pd.DataFrame(
                        {'d': [d], 'p': [p], 'q': [q], 'MAE': [mae], 'MAPE': [mape], 'mse': [mse], 'r2_score': [r2]})
                    df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv('grid_search_' + city + '.csv', index=False)
        #should choose the one with smallest error (and biggest R2) but also with simplest model
        return df

    def best_par(file):
        df = pd.read_csv(file)
        min_MAPE_row = df.loc[df['MAPE'].idxmin()]
        max_R2_row = df.loc[df['r2_score'].idxmax()]
        print(f"Best combination for max r2_score is with p: {max_R2_row['p']}, d: {max_R2_row['d']}, q: {max_R2_row['q']} ")
        #in Milano it's the same combination having min MAPE
        return min_MAPE_row['p'], min_MAPE_row['d'], min_MAPE_row['q']

    def learning_grid_search(df, p_best, d_best, q_best, strategy, city):
        df = df.set_index('date')

        first_day = datetime(2017, 10, 1)
        samples_train = 24*7*3 #3 weeks
        samples_test = 24*7 #1 week

        train_end = first_day + pd.to_timedelta(samples_train, unit='h')
        test_end = train_end + pd.to_timedelta(samples_test, unit='h')
        train = df.loc[first_day:train_end, ['number']]
        tr_values = train.values.astype(float)
        test = df.loc[train_end:test_end, ['number']]

        history = [x for x in tr_values]
        predictions = []

        for t in range(0, samples_test+1):
            model = ARIMA(history, order=(p_best, d_best, q_best))
            model_fit = model.fit()
            forecast = model_fit.forecast()

            prediction = forecast[0]
            predictions.append(prediction)
            obs = test.iloc[t]['number']
            history.append([obs])
            if strategy == "sliding_window":
                history = history[1:] #dropping first sample
        plt.plot(predictions, label = 'predictions')
        plt.title(f"ARIMA MODEL in {city}, p={p_best}, d={d_best}, q={q_best}")
        plt.plot(np.array(test), label = 'true values')
        plt.legend()
        plt.savefig(f"ARIMA_{strategy}_{city}.png")
        plt.show()

        mae = mean_absolute_error(test.values, predictions)
        mape = mae / test['number'].mean() * 100
        mse = mean_squared_error(test.values, predictions)
        r2 = r2_score(test.values, predictions)
        return mae, mape, mse, r2



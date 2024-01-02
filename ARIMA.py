from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt
import pymongo as pm
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from Lab_arima import Arima_Lab as AR

client = pm.MongoClient('bigdatadb.polito.it',ssl=True,authSource='carsharing', username='ictts',password='Ict4SM22!',tlsAllowInvalidCertificates=True)
db = client['carsharing']

d1 = datetime(2017, 9, 30, 23, 59, 59)
start = int(d1.replace(tzinfo=timezone.utc).timestamp())
d2 = datetime(2017, 10, 30, 23, 59, 59)
end = int(d2.replace(tzinfo=timezone.utc).timestamp())

city = 'Milano'

stats_df = AR.mean_variance(db['PermanentBookings'], start, end, city)

outlier_thresh_up = int(stats_df['mean'][0] + 2 * stats_df['variance'][0])
outlier_thresh_down = 60 * 5  # 5 minutes

outliers_removed_df = AR.remove_outliers(db['PermanentBookings'], start, end, city, outlier_thresh_down, outlier_thresh_up)

merged_df = AR.processing(outliers_removed_df)

AR.plot_n_cars(merged_df, city)

diff_data = AR.diff_plot(merged_df, city)

n_lags = 24

acf = AR.autocorr(diff_data, n_lags, city)

p_acf = AR.partial_autocorr(diff_data, n_lags, city)


# Use only the values correlated with the others for the grid search
acf_thresh = AR.filter(acf, 0.3)
p_acf_thresh = AR.filter(p_acf, 0.15)

acf_values = acf_thresh[np.argsort(acf_thresh)]
p_acf_values = p_acf_thresh[np.argsort(p_acf_thresh)]

samples_train = 24*14 # 2 weeks
samples_test = samples_train / 2 # 1 week

train, test = AR.train_test(diff_data, samples_train, samples_test)



#Simple ARIMA model
AR.simple_ARIMA(train, test, 17, 0, 21, city)

#7a
#execute it first time, then comment
#result_df = AR.grid_search(train, test, acf_values, p_acf_values, city)

grid_search_file = 'grid_search_'+city+'.csv'
best_p, best_d, best_q = AR.best_par(grid_search_file)
print(f"Best p:{best_p}, best d: {best_d}, best q: {best_q} having smallest MAPE")

#7b e c(cambiare solo city all'inizio)
strategy = "expanding_window"
mae, mape, mse, r2_score = AR.learning_grid_search(diff_data, best_p, best_d, best_q, strategy, city)

print(f'MAE: %.3f', mae)
print(f'MAPE: %.3f', mape)
print(f'MSE %.3f', mse)
print(f'R2 %.3f', r2_score)


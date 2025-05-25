
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tqdm
from scipy import optimize as opt
from sklearn import metrics as skmetrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive, ForecasterSarimax
from skforecast.sarimax import Sarimax
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster, \
    backtesting_sarimax, grid_search_forecaster, grid_search_sarimax


df_prices = pd.read_csv(
    '../code/data/crypto.csv', 
    index_col='dt', 
    parse_dates=['dt'])
df_prices.drop('TON-USDT', axis=1, inplace=True)
df_prices.columns = [c.split('-')[0] for c in df_prices.columns]
print(df_prices.head())

days_shift = 7
df_returns = df_prices.diff(days_shift) / df_prices.shift(days_shift)
df_returns = df_returns[df_returns.isna().sum(axis=1) == 0]
df_returns = df_returns[df_returns.index >= '2022-01-01']

n_observations, n_assets = df_returns.shape
print(n_observations, n_assets)
print(df_returns.head())

threshold_date = '2023-10-01'
df_returns_test = df_returns[df_returns.index >= threshold_date]
df_returns_train = df_returns[df_returns.index < threshold_date]
print(df_returns_train.shape, df_returns_test.shape)

def mse_last_value(y_true, y_pred):
    idxs = range(0, len(y_true), days_shift)
    return skmetrics.mean_squared_error(
        y_true.iloc[idxs], y_pred.iloc[idxs])

# HP optimization
cv = TimeSeriesFold(
    steps=days_shift,
    initial_train_size=50,
    refit=False,
)

lags_grid = {
    '0': 1,
    '1': range(1, 4),
    '2': range(1, 8),
    '3': range(1, 15),
}

arima_params = {
    'order': [
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (2, 0, 2),
        (2, 1, 2),
    ]
}

model_grid = [
    (LinearRegression, 'LR', {}),
    (RandomForestRegressor, 'RF', {
        'n_estimators': [10, 50, 100], 
        'random_state': [27]
        }),
]

best_models = {}
for c in df_returns.columns:
    # ARMA
    train_data = df_returns_train[c].reset_index(drop=True)
    forecaster = ForecasterSarimax(
                regressor=Sarimax(),
                forecaster_id=f'ARIMA_{c}'
                )
    metric = grid_search_sarimax(
                forecaster=forecaster,
                y=train_data,
                param_grid=arima_params,
                cv=cv,
                metric=mse_last_value,
                return_best=True,
                n_jobs='auto',
                verbose=False,
                show_progress=True
            )
    if best_models.get('ARIMA') is None:
        best_models['ARIMA'] = []
    best_models['ARIMA'].append(forecaster)

    # ML
    for model_builder, model_name, params in model_grid:
        train_data = df_returns_train[c].reset_index(drop=True)
        forecaster = ForecasterRecursive(
                    regressor=model_builder(),
                    lags=range(1, 8),
                    forecaster_id=f'{model_name}_{c}'
                )
        metric = grid_search_forecaster(
                      forecaster=forecaster,
                      y=train_data,
                      param_grid=params,
                      lags_grid=lags_grid,
                      cv=cv,
                      metric=mse_last_value,
                      return_best=True,
                      n_jobs='auto',
                      verbose=False,
                      show_progress=True
                  )
        if best_models.get(model_name) is None:
            best_models[model_name] = []
        best_models[model_name].append(forecaster)


print(best_models.keys())

# evaluate models on test data
cv = TimeSeriesFold(
    steps=days_shift,
    initial_train_size=50,
    refit=True,
)
backtest_metrics = {}

for model_type, models in best_models.items():
    if model_type == 'ARIMA':
        for c, forecaster in zip(df_returns_test.columns, models):
            data_test = df_returns_test[c].reset_index(drop=True)
            metric, predictions = backtesting_sarimax(
                          forecaster=forecaster,
                          y=data_test,
                          cv=cv,
                          metric=mse_last_value,
                          n_jobs=-1,
                      )
            if backtest_metrics.get(model_type) is None:
                backtest_metrics[model_type] = []
            backtest_metrics[model_type].append(metric.values.item())
    else:
        for c, forecaster in zip(df_returns_test.columns, models):
            data_test = df_returns_test[c].reset_index(drop=True)
            metric, pred = backtesting_forecaster(
                forecaster=forecaster,
                y=data_test,
                cv=cv,
                metric=mse_last_value,
                n_jobs=-1,
            )
            if backtest_metrics.get(model_type) is None:
                backtest_metrics[model_type] = []
            backtest_metrics[model_type].append(metric.values.item())
            

# martingal mse
backtest_metrics['MARTINGAL'] = (
    (df_returns_test - df_returns_test.shift())**2
    ).mean(axis=0).to_list()
backtest_metrics['NAIVE'] = (
    (df_returns_train.mean(axis=0) - df_returns_test)**2
    ).mean(axis=0).to_list()

# mse on test data
(pd.DataFrame(
    {m: backtest_metrics[m] for m in [
        'NAIVE', 'MARTINGAL', 'LR', 'ARIMA', 'RF'
        ]}, 
    index=df_returns.columns) * 1000
    ).to_latex('../tables/ml_eval_metrics.tex',
               caption='Качество прогнозирования',
               float_format='%.2f',
               position='h',
               label='tab:ml_eval_metrics'
               )

def portfolio_optimizer(mu_hat, cov_hat, tau):
    def objective(w):
        w = w.reshape((-1, 1))
        return (w.T @ cov_hat @ w - tau * w.T @ mu_hat).item()
        
    def unit_portfolio(w):
        return np.abs(w).sum() - 1
        
    eq_cons = {
        'type': 'eq',
        'fun': unit_portfolio,
    }
    bounds = [(-1, 1) for i in range(n_assets)]
    x0 = np.ones(n_assets) / n_assets
    sol = opt.minimize(
        fun=objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[eq_cons]
    )
    if sol.success:
        return sol.x

# %%
def frontier_evaluator(mu_hat, cov_hat, ret_true, frontier_tau):
    frontier = np.full_like(frontier_tau, np.nan)
    for i in range(len(frontier_tau)):
        tau = frontier_tau[i]
        w = portfolio_optimizer(mu_hat, cov_hat, tau)
        if w is None:
            print('not converged')
            continue
        roi = np.dot(w, ret_true)
        frontier[i] = roi
    return frontier

# mu estimators
def naive_estimator(df_hist):
    return df_hist.mean(axis=0)

def martingal_estimator(df_hist):
    return df_hist.iloc[-1]

def ml_estimator_builder(models):
    def func(df_hist):
        mu_hat = []
        for c, forecaster in zip(df_hist.columns, models):
            y = df_hist[c].reset_index(drop=True)
            forecaster.fit(y)
            mu_hat.append(forecaster.predict(days_shift).iloc[-1])
        return np.array(mu_hat)
    return func

n_assets = df_returns.shape[1]

idx_most_risky = np.argmax(df_returns_train.describe().T['std'])
idx_less_risky = np.argmin(df_returns_train.describe().T['std'])
idx_best_return = np.argmax(df_returns_train.describe().T['mean'])
idx_worst_return = np.argmin(df_returns_train.describe().T['mean'])
print(idx_most_risky, 
      idx_less_risky, 
      idx_best_return, 
      idx_worst_return)

def single_asset_portfolio_builder(idx):
    w =  np.zeros(n_assets)
    w[idx] = 1
    return w

def uniform_portfolio_builder():
    return np.full(n_assets, 1 / n_assets)

print(best_models.keys())

results_frontier = []
results_trivial = []

min_history_leng = 91
total_runs = df_returns_test.shape[0] - days_shift - min_history_leng
print(total_runs)
frontier_tau = np.linspace(0.01, 1, 5)

mu_estimators = [
    ('NAIVE', naive_estimator),
    ('MARTINGAL', martingal_estimator),
    ('LR', ml_estimator_builder(best_models['LR'])),
    ('ARIMA', ml_estimator_builder(best_models['ARIMA'])),
    ('RF', ml_estimator_builder(best_models['RF'])),
]

trivial_portfolios = [
    ('UNIFORM', uniform_portfolio_builder()),
    ('MOST_RISKY', single_asset_portfolio_builder(idx_most_risky)),
    ('LESS_RISKY', single_asset_portfolio_builder(idx_less_risky)),
    ('BEST_RETURN', single_asset_portfolio_builder(idx_best_return)),
    ('WORST_RETURN', single_asset_portfolio_builder(idx_worst_return)),
]

for t in tqdm.trange(total_runs):
    # prepare data
    idx_history = min_history_leng + t
    idx_future = idx_history + days_shift - 1
    df_history = df_returns_test.iloc[:idx_history]
    df_future = df_returns_test.iloc[idx_future]

    # estimate cov, common for all models
    cov_hat = df_history.cov().values

    # estimate mu using list of models
    frontiers = []
    for name, mu_estimator in mu_estimators:
        mu_hat = mu_estimator(df_history)
        
        # evaluate each portfolio in frontier for currnet model
        roi_frontier = frontier_evaluator(
            mu_hat, cov_hat, df_future.values, frontier_tau)
        frontiers.append(roi_frontier)
    results_frontier.append(frontiers)
    
    # evaluate trivial strategies 
    trivials = []
    for name, w in trivial_portfolios:
        roi = np.dot(w, df_future.values).item()
        trivials.append(roi)
    results_trivial.append(trivials)

frontier_means = np.nanmean(results_frontier, axis=0)
frontier_stds = np.nanstd(results_frontier, axis=0)

trivial_means = np.mean(results_trivial, axis=0)
trivial_stds = np.std(results_trivial, axis=0)

print(frontier_means, frontier_stds)

fig, ax = plt.subplots()
for m, s, (label, _) in zip(
    frontier_means, frontier_stds, mu_estimators):
    ax.plot(s, m, marker='o', label=label)

ax.scatter(trivial_stds, trivial_means, color='grey')
for s, m, (name, _) in zip(
    trivial_stds, trivial_means, trivial_portfolios):
    ax.text(s + 0.003, m, name)
    
ax.set_xlabel('ROI std')
ax.set_ylabel('ROI mean')
ax.set_xlim(0, 0.2)
ax.legend()
fig.savefig('../images/result_frontiers.png')

print(np.isnan(results_frontier).mean(axis=0))

# mean ROI
(pd.DataFrame(
    frontier_means, columns=[f'{t: .2f}' for t in frontier_tau], 
    index=[n for n, _ in mu_estimators]) * 1000
    ).to_latex('../tables/roi_mean.tex',
            caption='Средние ROI $\cdot 10^3$',
            float_format='%.4f',
            position='h',
            label='tab:roi_mean',
            )

# std ROI
(pd.DataFrame(
    frontier_stds, columns=[f'{t: .2f}' for t in frontier_tau], 
    index=[n for n, _ in mu_estimators]) * 100
    ).to_latex('../tables/roi_std.tex',
            caption='Стандартное отклонение ROI $\cdot 10^2$',
            float_format='%.4f',
            position='h',
            label='tab:roi_std'
            )

(pd.DataFrame({
    'mean ROI $\cdot 10^3$': trivial_means * 1000,
    'std ROI $\cdot 10^2$': trivial_stds * 100,
    }, 
    index=[n.replace('_', ' ') for n, _ in trivial_portfolios])
).to_latex('../tables/trivial_rois.tex',
            caption='Тривиальные портфели',
            float_format='%.4f',
            position='h',
            label='tab:trivial_rois',
            )

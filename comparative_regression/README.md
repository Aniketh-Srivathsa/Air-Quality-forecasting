# Smart city air-quality prediction

An executable adaptation of Ameer et al., “Comparative Analysis of Machine
Learning Techniques for Predicting Air Quality in Smart Cities” (IEEE Access,
2019). It compares Decision Tree, Random Forest, Multilayer Perceptron, and
Gradient Boosting regressors for next-hour PM2.5 prediction, reporting MAE,
RMSE, and R² on a chronological holdout. The best validation model is refit on
the available history and used for a recursive forecast.

## Data source and limits

The original study evaluates five Chinese cities using hourly records from
2010–2015. Those old observations are not bundled here. This adaptation fetches
recent hourly PM2.5 and meteorological series for those same cities from the
keyless Open-Meteo Forecast and Air Quality APIs. The API provides modelled
estimates and forecasts, not certified sensor readings. Forecasts further than
one hour recurse on the model’s own PM2.5 estimates, so uncertainty grows with
the horizon. This is a current-data demonstration, not an exact replication of
the paper’s historical experiment or reported scores.

## Run

Requires Python 3.10+ and internet access.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m air_quality --city beijing --hours 24
```

Cities supported: `beijing`, `chengdu`, `guangzhou`, `shanghai`, `shenyang`.
The default is Beijing. The forecast horizon can be 1–119 hours.

The command prints each model’s chronological validation metrics and the
selected model’s forecast. It writes the fitted model to `models/` and the
comparison and hourly forecast to `reports/`.

## Method

The target is PM2.5 at hour *t+1*. Predictors at hour *t* include current PM2.5,
temperature, relative humidity, pressure, wind speed/direction, precipitation,
calendar fields, and lag/rolling PM2.5 values. An 80/20 chronological split
avoids random shuffling of the time series. Missing predictor values are
median-imputed within each model pipeline.

## Reference

Ameer, S. et al. (2019). “Comparative Analysis of Machine Learning Techniques
for Predicting Air Quality in Smart Cities.” *IEEE Access*, 7, 128325–128338.
https://doi.org/10.1109/ACCESS.2019.2925082


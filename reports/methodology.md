# Methodology comparison and deviations

Reference: S. V. Belavadi, S. Rajagopal, R. Ranjani, and R. Mohan, “Air Quality Forecasting using LSTM RNN and Wireless Sensor Networks,” *Procedia Computer Science*, vol. 170, pp. 241–248 (2020), DOI [10.1016/j.procs.2020.03.036](https://doi.org/10.1016/j.procs.2020.03.036).

## Paper-to-project comparison

| Topic | Paper | This implementation | Status / deviation |
|---|---|---|---|
| Data sources | Authors' Bengaluru WSN and Indian government hourly AQI/API data; Amaravati APPCB data used in results | OpenCity Jayanagar 5th Block KSPCB CSV, 15-minute rows from 2024–25 | **Major substitution** in period, location/source and sampling. Not original data. |
| Sampling | Pollutant averages are hourly; WSN nodes send minute-level data to EC2, aggregated hourly | Numeric pollutant values averaged into clock-hour bins | Hourly target cadence follows paper; input granularity differs. |
| Weather | Paper explicitly excludes weather as a direct forecasting input | Weather-like columns are filtered from candidate pollutants and not passed to model | Followed. |
| Pollutant modeling | Individual pollutant prediction models | One univariate model per selected pollutant | Followed. |
| Outlier handling | Paper describes removing values whose absolute Z-score is greater than 3 | Source pollutant measurements exceeding absolute population Z-score 3 are set missing before hourly aggregation | Z-score population convention is an implementation choice; pre-aggregation placement follows the requested pipeline. |
| Missing data | Paper describes imputation with the value from one week earlier; mean used where that is unavailable | Shift by 168 hourly positions, then series mean | Followed at hourly frequency. |
| Network | TensorFlow LSTM 64 ReLU with return-sequences, LSTM 32 ReLU, Dense 1 linear | Same Keras architecture | Followed. |
| Objective / optimizer | MSE loss and Adam | MSE loss and Adam | Followed. |
| Split | 67% training, remainder test; sequence description says final 33% is testing | Chronological first 67% / final 33%, no shuffle | Follows the specified ratio and preserves time order. Paper does not state shuffle details. |
| Lookback | Not specified in paper | 24 hourly observations | **Assumption**; surfaced in UI and can be changed. |
| Training duration / stopping | Paper discusses training but does not specify maximum epoch count or stopping rule in the reported methodology | Up to 50 epochs with early stopping; validation is the final 10% of training sequences; patience defaults to 5 | **Implementation choice** required by this project request. |
| Scaling | No scaling procedure specified in the paper | MinMaxScaler fit only on training observations; inverse transformed before metrics | **Implementation choice** to stabilize optimization without test leakage. |
| Forecast horizon | Forecasts next five hours described | Five recursively fed predictions | Horizon follows paper; recursive feedback is explicitly requested here. |
| Metrics | RMSE and RMSE as percentage of pollutant maximum during observed period | Test RMSE and 100 × RMSE / maximum actual test value | Requested test-set definition is used; denominator window therefore differs if paper's full observation period is meant. Units remain source units, not paper ppm. |
| Hardware/platform | Arduino Uno, MQ-2, MQ-135, Sharp dust sensor, GSM, EC2 server, APIs and web front end | No hardware or cloud ingestion; local dashboard only | Conceptual diagram only; explicitly not implemented. |

## Assumptions and safe behavior

- Timestamps that cannot parse are discarded. Duplicate exact timestamps retain the last row before hourly aggregation.
- A station selector is shown when the source contains a station field. Station identifiers are not treated as measurements.
- Pollutant columns are inferred from numeric-valued columns after excluding timestamp, station identifiers, and obvious weather/metadata names (including common abbreviations AT/RH/WS/WD/RF/SR/BP). This accommodates source schema variation, but users should inspect the detected list.
- The hourly grid spans first to last valid timestamp; unobserved hours become missing and are imputed by the weekly rule/mean.
- If a series has no observed values, preprocessing raises a clear error. If there are too few observations for a split plus lookback, training stops with a clear error.
- Percentage RMSE is undefined (NaN in internal result; shown as undefined in UI) if maximum actual test value is zero.

## Interpretation

This is a runnable reproduction of the core model pattern and requested processing pipeline, not an exact reproduction of the paper experiment. Different data, sensor source, measurement coverage, units, lookback assumption, scaling and stopping configuration prevent direct comparison to or claims of matching the published numerical results.

## Current-date data mode (additional deviation)

The dashboard also offers an optional current-data path through the Open-Meteo Air Quality API, using hourly CAMS model estimates at approximate Jayanagar coordinates (12.925, 77.583). The API is queried for up to 92 past days and one forecast day; only completed timestamps no later than the latest completed UTC hour are retained as model history. API-forward forecast hours are excluded. The same Z-score/hourly/imputation pipeline, univariate architecture, chronological 67/33 split and five-step recursive forecast then apply to that returned history.

This live mode is an additional substitute source and is not part of the paper's experiment. CAMS is a gridded atmospheric model product and may be spatially coarse; it does not reproduce station-grade readings from Bengaluru WSN or government instruments. Live-mode test metrics therefore compare predictions with held-out CAMS estimates, not independently verified sensor observations. Open-Meteo/CAMS attribution is displayed in the app. Pollutant fields that the API omits are handled through the existing detected-column workflow; fewer than 72 completed hourly rows produces a clear error. Source modes are kept separate to avoid joining measurements and model estimates.

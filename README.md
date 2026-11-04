# Bike Sharing Monitoring

## Launch instructions

Run `docker compose up -d`. To rebuild after code changes, use `docker compose up -d --build`.

Open http://localhost:8000, then go to Bike Rental Monitoring > Reports to see the results.
The exported drift report is in `reports/diagnostic_report.html`.

## 1. Performance analysis

W3 (February 15–21, 2011) showed the worst performance compared to W1 and W2. Here are the results:

| Week | MAE | RMSE |
| --- | ---: | ---: |
| W1 | 14.10 | 22.27 |
| W2 | 14.29 | 21.25 |
| W3 | 25.41 | 38.42 |

W3 has the highest RMSE, so it was selected for the drift analysis.

## 2. Drift analysis

W3 was warmer, less humid, and windier than January, while higher hourly rental counts became more common. The model underpredicted demand, especially during temperatures outside its training range. Changed weather conditions could therefore explain part of the performance drop, but the drift results alone don't prove causation or concept drift.

## 3. Remediation

The model was trained on earlier January data, which didn't cover some of the warmer conditions seen in W3. Its predictions didn't keep up with the actual demand.

I would automatically trigger retraining with recent labeled data if prediction error stays above an agreed threshold across several monitoring windows. I would check data quality first, then test the new model on later held-out data before replacing the current one. Drift alone wouldn't be enough to replace it.

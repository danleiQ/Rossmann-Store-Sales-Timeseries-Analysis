# Rossman-Store-Sales-Timerseries-Analysis
## Objection
- Perform Time Series Analysis (seasonal decomposition, trends, autocorrelation) on Sales Data.
- Predict future sales with Holt-Winters, Multilinear Regression, ARMA models.
## Datasets
Data t were collected from Kaggle.com,the link is as follow:
https://www.kaggle.com/c/rossmann-store-sales/data 
(in which only the ‘train.csv‘was used.)
## GPAC
![image](https://github.com/danleiQ/Rossmann-Store-Sales-Timeseries-Analysis/blob/main/GPAC.jpg)
## Summary
|  Models| Mean of errors | Variance of errors | MSE | RMSE | MAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| Holt-Winter | 721.6277 | 1984371.4182 | 2505118.0101 | 1582.7565 | 0.2944 |
| Multiple linear Regression | -126.5149 | 69036.4002| 85042.4161| 291.6203 | 0.0508 |
|ARMA(3,2)| -68.4986 | 70055.3076 | 74747.3687 | 273.3997 | 0.0540 |
|ARMA(1,1)| 49.8397 | 18728.0066 | 21211.9981 | 145.6434 | 0.0239 |






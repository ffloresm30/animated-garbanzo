from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import joblib



url = "https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv"

df = pd.read_csv(url)
df.head()

df["date"] = pd.to_datetime(df['date'])
df.set_index("date", inplace=True)
ts = df["sales"]
ts.head()

fig, axis = plt.subplots(figsize=(10,5))
sns.lineplot(data=ts)
plt.title("Serie Temporal de Ventas")
plt.tight_layout()
plt.show()

decomposition = seasonal_decompose(ts, period=12)
trend= decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#Grafica de tendencia

fig, axis= plt.subplots(figsize=(10,5))
sns.lineplot(data=ts, label="Original")
sns.lineplot(data=trend, label="Tendencia")
plt.title("Analisis de la Tendencia")
plt.tight_layout()
plt.show()

def test_stationarity(timeseries):
    print("Resultados de la prueba de Dickey-Fuller:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Estadístico de Prueba", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Valor Crítico (%s)" % key] = value
    print(dfoutput)

test_stationarity(ts.dropna())

fig, axis = plt.subplots(figsize=(10, 5))
sns.lineplot(data=residual, label="Residuos")
plt.title("Análisis de la Variabilidad")
plt.tight_layout()

model = auto_arima(ts, seasonal=True, m=12, trace=True)
print(model.summary())

forecast = model.predict(n_periods=12)
fig, axis = plt.subplots(figsize=(10, 5))
sns.lineplot(data=ts, label="Original")
sns.lineplot(data=forecast, label="Predicción", color="green")
plt.title("Predicción con ARIMA")
plt.tight_layout()
plt.show()


joblib.dump(model, "arima_model.pkl")
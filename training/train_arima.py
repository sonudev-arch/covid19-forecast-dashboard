from utils.arima_model import SimpleARIMA
import pickle
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
def train():
        
    # Load your data (same as before)
    merged = pd.read_csv("datasets/merged_covid_vacc.csv", parse_dates=["Date"], index_col="Date")

    series = merged['NewConfirmed_7d']

    TRAIN_CUTOFF = "2021-06-01"
    FORECAST_DAYS = 30

    train = series[series.index < TRAIN_CUTOFF]
    val   = series[series.index >= TRAIN_CUTOFF]

    model = SimpleARIMA()
    model.fit(series.values)
    with open("arima_model.sav", "wb") as f:
        pickle.dump({
            "model": model,
            "last_date": series.index[-1]
        }, f)

    print("Model retrained and saved successfully.")

if __name__ == "__main__":
    train()

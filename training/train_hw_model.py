from utils.holt_winters import HoltWinters
from utils.data_loader import load_merged
import joblib
import json

series = load_merged()['NewConfirmed_7d']

# Split
train = series.iloc[:-30]
val   = series.iloc[-30:]

# Train
model = HoltWinters(m=7)
model.fit(train.values)

# Save
joblib.dump(model, "models/holt_winters_model.joblib")

joblib.dump({
    "series_len": len(series),
    "last_date": str(series.index[-1])
}, "models/holt_winters_meta.joblib")
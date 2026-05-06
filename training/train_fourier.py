if __name__ == "__main__":
    import json
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    # load data
    df = pd.read_csv("../datasets/merged_covid_vacc.csv")
    series = df["NewConfirmed_7d"]

    # define model
    def wave_model(t, A1, omega1, phi1,
                    A2, omega2, phi2,
                    A3, omega3, phi3,
                    C, decay):
        damp = np.exp(-decay * t)
        return (C * damp
                + A1 * np.sin(omega1*t + phi1) * damp
                + A2 * np.sin(omega2*t + phi2) * damp
                + A3 * np.sin(omega3*t + phi3) * np.exp(-decay*0.5*t))

    # fit
    t = np.arange(len(series), dtype=float)
    y = series.values.astype(float)

    popt, _ = curve_fit(wave_model, t, y, maxfev=50000)

    # save
    state = {
        "params": popt.tolist(),
        "series_len": len(series),
        "last_date": str(pd.to_datetime(df["Date"]).iloc[-1].date())
    }

    with open("../models/fourier_model.json", "w") as f:
        json.dump(state, f, indent=2)

    print("Model saved successfully ✅")
    
        

import requests
import pandas as pd


def get_historical_data(coin: str = "ethereum", days: int = 120) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.set_index("timestamp")
    df = df.sort_index()

    return df

import time

import pandas as pd
import requests


def get_historical_data(
    coin: str = "ethereum",
    days: int = 120,
    vs_currency: str = "usd",
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                raise RuntimeError("Rate limit atingido na API CoinGecko.")

            response.raise_for_status()
            data = response.json()

            if "prices" not in data:
                raise ValueError("Resposta da API sem campo obrigatório: prices")

            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])

            if "total_volumes" in data:
                volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
                df = prices.merge(volumes, on="timestamp", how="left")
            else:
                df = prices.copy()
                df["volume"] = 0.0

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()

            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            df = df.ffill().bfill()

            if df.empty:
                raise ValueError("DataFrame vazio após coleta.")

            return df[["price", "volume"]]

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"Falha ao coletar dados da CoinGecko: {last_error}")

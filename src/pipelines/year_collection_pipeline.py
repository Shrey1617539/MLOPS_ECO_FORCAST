import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_cities
# from src.logging import get_logger
from src.data.preprocessing import WeatherDataPreprocessor

# # logger = get_logger(__name__)

class YearCollectionPipeline:
    """One‑time pull of 00:00 UTC data for 2024, raw + processed."""

    HISTORY_URL = "https://history.openweathermap.org/data/2.5/history/city"

    def __init__(self):
        self.cities = get_cities()
        self.year = 2024
        root = os.getenv("DATA_ROOT", "data")
        self.raw_dir = os.path.join(root, "raw")
        self.preprocessor = WeatherDataPreprocessor()
        os.makedirs(self.raw_dir, exist_ok=True)

    def run(self):
        for city in self.cities:
            # logger.info(f"=== collecting {city['name']} for {self.year} ===")
            self._collect_city(city)

    def _collect_city(self, city):
        slug = city["name"].lower().replace(" ", "_")
        raw_fp = os.path.join(self.raw_dir, f"{slug}.csv")

        # rolling one‑year window May 1, 2024 → Apr 29, 2025
        start_dt = datetime(self.year, 5, 1, tzinfo=timezone.utc)
        end_dt   = datetime(self.year + 1, 4, 29, tzinfo=timezone.utc)
        dt = start_dt

        while dt <= end_dt:
            try:
                ts = int(dt.timestamp())
                params = {
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "type": "hour",
                    "start": ts,
                    "cnt": 1,
                    "appid": '3b1d6cea391a2c702c99b5e77995a2c2',
                    "units": "metric",
                }
                resp = requests.get(self.HISTORY_URL, params=params, timeout=30)
                resp.raise_for_status()
                recs = resp.json().get("list", [])
                if not recs:
                    # logger.warning(f"no data for {slug} @ {dt.date()}")
                    dt += timedelta(days=1)
                    continue

                hr = recs[0]
                # build record matching data_collection format
                record = {
                    "city": city["name"],
                    "country": city.get("country", ""),
                    "date": dt.strftime("%Y-%m-%d"),
                    "temperature": hr["main"]["temp"],
                    "humidity": hr["main"]["humidity"],
                    "pressure": hr["main"]["pressure"],
                    "wind_speed": hr["wind"]["speed"],
                    "weather_main": hr.get("weather", [{}])[0].get("main", ""),
                    "weather_description": hr.get("weather", [{}])[0].get("description", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                df_day = pd.DataFrame([record])
                write_header = not os.path.exists(raw_fp)
                df_day.to_csv(raw_fp, mode="a", index=False, header=write_header)
                # logger.info(f"saved raw {slug} @ {record['date']}")

                # now process this date
                self.preprocessor.preprocess_city_data(slug, record["date"])

            except Exception as e:
                print(f"Error processing {city['name']} on {dt.strftime('%Y-%m-%d')}: {e}")
                # logger.error(f"{city['name']} {dt.strftime('%Y-%m-%d')} error: {e}")

            dt += timedelta(days=1)


if __name__ == "__main__":
    YearCollectionPipeline().run()
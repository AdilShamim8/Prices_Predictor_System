"""
sample_predict.py
-----------------
Send a single-record prediction request to the deployed MLflow model server.

Usage:
    python sample_predict.py
    python sample_predict.py --url http://127.0.0.1:8000/invocations
"""
import argparse
import json
import logging
import sys

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample input — replace values with an actual property from the dataset
# ---------------------------------------------------------------------------
SAMPLE_INPUT = {
    "dataframe_records": [
        {
            "Order": 1,
            "PID": 5286,
            "MS SubClass": 20,
            "Lot Frontage": 80.0,
            "Lot Area": 9600,
            "Overall Qual": 5,
            "Overall Cond": 7,
            "Year Built": 1961,
            "Year Remod/Add": 1961,
            "Mas Vnr Area": 0.0,
            "BsmtFin SF 1": 700.0,
            "BsmtFin SF 2": 0.0,
            "Bsmt Unf SF": 150.0,
            "Total Bsmt SF": 850.0,
            "1st Flr SF": 856,
            "2nd Flr SF": 854,
            "Low Qual Fin SF": 0,
            "Gr Liv Area": 1710.0,
            "Bsmt Full Bath": 1,
            "Bsmt Half Bath": 0,
            "Full Bath": 1,
            "Half Bath": 0,
            "Bedroom AbvGr": 3,
            "Kitchen AbvGr": 1,
            "TotRms AbvGrd": 7,
            "Fireplaces": 2,
            "Garage Yr Blt": 1961,
            "Garage Cars": 2,
            "Garage Area": 500.0,
            "Wood Deck SF": 210.0,
            "Open Porch SF": 0,
            "Enclosed Porch": 0,
            "3Ssn Porch": 0,
            "Screen Porch": 0,
            "Pool Area": 0,
            "Misc Val": 0,
            "Mo Sold": 5,
            "Yr Sold": 2010,
        }
    ]
}


def predict(url: str) -> None:
    """Send a prediction request and print the result.

    Args:
        url: Full URL of the MLflow model server invocations endpoint.

    Raises:
        SystemExit: On network error or non-200 HTTP response.
    """
    logger.info("Sending prediction request to: %s", url)

    headers = {"Content-Type": "application/json"}
    payload = json.dumps(SAMPLE_INPUT)

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
    except requests.exceptions.ConnectionError:
        logger.error(
            "Could not connect to %s. Is the MLflow prediction server running? "
            "Start it with: python run_deployment.py",
            url,
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.error("Request to %s timed out after 15 seconds.", url)
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected request error: %s", exc)
        sys.exit(1)

    if response.status_code == 200:
        prediction = response.json()
        logger.info("Prediction successful.")
        print("\nPrediction result:")
        print(json.dumps(prediction, indent=2))
    else:
        logger.error(
            "Server returned HTTP %d:\n%s", response.status_code, response.text
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a sample prediction request to the deployed MLflow model server."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/invocations",
        help="MLflow model server endpoint (default: http://127.0.0.1:8000/invocations)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(url=args.url)

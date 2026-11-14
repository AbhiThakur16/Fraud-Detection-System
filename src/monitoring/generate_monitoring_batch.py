
import random
import time
import requests

from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

TOTAL_TRANSACTIONS = 200

random.seed(42)


def generate_transaction(index):

    amount = round(random.uniform(100, 15000), 2)
    hour = random.randint(0, 23)
    account_count = random.randint(5, 100)
    previous_avg = round(random.uniform(500, 5000), 2)

    transaction_type = random.choice(
        ["transfer", "purchase", "withdrawal"]
    )

    channel = random.choice(
        ["upi", "mobile", "web"]
    )

    device = random.choice(
        ["mobile", "desktop", "tablet"]
    )

    is_weekend = random.choice([0, 1])
    is_night = int(hour <= 5 or hour >= 22)

    previous_max = round(
        previous_avg * random.uniform(1.2, 3.0), 2
    )

    transaction = {
        "amount": amount,
        "transaction_type": transaction_type,
        "transaction_channel": channel,
        "currency": "INR",
        "transaction_status": "success",
        "device_type": device,
        "location": random.choice(
            ["Gurugram", "Delhi", "Noida", "Mumbai", "Jaipur"]
        ),

        "account_age_days": random.randint(30, 2000),
        "transaction_hour": hour,
        "transaction_day": random.randint(1, 28),
        "transaction_day_of_week": random.randint(0, 6),
        "transaction_month": random.randint(1, 12),

        "is_weekend": is_weekend,
        "is_night_transaction": is_night,
        "is_high_value": int(amount >= 10000),

        "log_amount": round(__import__("math").log1p(amount), 4),

        "is_purchase": int(transaction_type == "purchase"),
        "is_transfer": int(transaction_type == "transfer"),
        "is_merchant_transaction": int(
            transaction_type == "purchase"
        ),

        "is_rapid_transaction": random.choice([0, 1]),

        "account_transaction_count": account_count,
        "account_transactions_1h": random.randint(0, 8),
        "account_transactions_24h": random.randint(1, 20),

        "account_previous_avg_amount": previous_avg,
        "account_previous_max_amount": previous_max,

        "amount_deviation_ratio": round(
            amount / max(previous_avg, 1), 4
        ),

        "has_previous_account_history": 1,

        "merchant_previous_count": random.randint(0, 100),
        "account_merchant_previous_count": random.randint(0, 20),

        "account_previous_24h_avg_amount": round(
            random.uniform(300, 5000), 2
        )
    }

    return transaction


def main():

    print("=" * 55)
    print("SYNTHETIC MONITORING BATCH GENERATOR")
    print("=" * 55)

    successful = 0
    failed = 0

    for index in range(1, TOTAL_TRANSACTIONS + 1):

        transaction = generate_transaction(index)

        try:

            response = requests.post(
                API_URL,
                json={"transaction": transaction},
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                successful += 1

                prediction = result.get("prediction", {})

                print(
                    f"[{index}/{TOTAL_TRANSACTIONS}] "
                    f"Saved | Amount: ₹{transaction['amount']} | "
                    f"Prediction: {prediction.get('status', 'N/A')}"
                )

            else:

                failed += 1

                print(
                    f"[{index}/{TOTAL_TRANSACTIONS}] "
                    f"Failed: {response.status_code} "
                    f"{response.text[:200]}"
                )

        except requests.RequestException as error:

            failed += 1
            print(f"Request error: {error}")

        time.sleep(0.05)

    print("\n" + "=" * 55)
    print("BATCH GENERATION COMPLETED")
    print("=" * 55)

    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    print(f"Completed at: {datetime.now()}")

    print("\nCheck PostgreSQL prediction_history table.")
    print("Then run the Data Drift Monitoring API.")


if __name__ == "__main__":
    main()

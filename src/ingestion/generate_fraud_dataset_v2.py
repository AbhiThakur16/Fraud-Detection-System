
import os
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

RANDOM_SEED = 42

TOTAL_TRANSACTIONS = 50000
TOTAL_FRAUD = 1000
TOTAL_NORMAL = TOTAL_TRANSACTIONS - TOTAL_FRAUD

OUTPUT_PATH = "data/processed/fraud_transactions_v2.csv"

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_database_engine():

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )

    return create_engine(connection_url)


# ==========================================
# LOAD EXISTING ACCOUNTS AND MERCHANTS
# ==========================================

def load_reference_data():

    print("\nConnecting to PostgreSQL...")

    engine = get_database_engine()

    try:

        accounts = pd.read_sql_query(
            "SELECT account_id FROM accounts",
            engine
        )["account_id"].to_numpy()

        merchants = pd.read_sql_query(
            "SELECT merchant_id FROM merchants",
            engine
        )["merchant_id"].to_numpy()

        print(f"Accounts loaded: {len(accounts)}")
        print(f"Merchants loaded: {len(merchants)}")

        if len(accounts) < 2 or len(merchants) == 0:
            raise ValueError("Insufficient reference data.")

        return accounts, merchants

    finally:

        engine.dispose()


# ==========================================
# TRANSACTION GENERATOR
# ==========================================

class FraudDatasetGenerator:

    def __init__(self, accounts, merchants):

        self.accounts = accounts
        self.merchants = merchants

        self.account_baseline = {
            account: rng.uniform(250, 1200)
            for account in accounts
        }

        self.account_set = set(accounts)

        self.records = []

        self.start_time = (
            datetime.now(timezone.utc) -
            timedelta(days=90)
        )

        self.transaction_id = 1

        self.channels = [
            "upi",
            "card",
            "net_banking",
            "atm",
            "mobile",
            "web"
        ]

        self.transaction_types = [
            "purchase",
            "transfer",
            "withdrawal",
            "deposit",
            "refund"
        ]

        self.statuses = [
            "settled",
            "authorized",
            "declined"
        ]

    def random_time(self):

        seconds = int(rng.integers(0, 90 * 24 * 60 * 60))

        return self.start_time + timedelta(seconds=seconds)

    def random_account(self):

        return rng.choice(self.accounts)

    def random_merchant(self):

        return int(rng.choice(self.merchants))

    def create_transaction(
        self,
        account,
        amount,
        occurred_at,
        transaction_type="purchase",
        channel=None,
        fraud=False,
        scenario="normal",
        merchant_id=None
    ):

        if channel is None:
            channel = rng.choice(self.channels)

        receiver_account = None

        if transaction_type == "transfer":

            receiver_account = self.random_account()

            while receiver_account == account:
                receiver_account = self.random_account()

        if merchant_id is None and transaction_type in [
            "purchase",
            "refund"
        ]:
            merchant_id = self.random_merchant()

        record = {

            "transaction_id": self.transaction_id,

            "sender_account_id": int(account),

            "receiver_account_id": (
                int(receiver_account)
                if receiver_account is not None
                else None
            ),

            "merchant_id": merchant_id,

            "transaction_type": transaction_type,

            "transaction_channel": channel,

            "amount": round(float(amount), 2),

            "currency": "INR",

            "transaction_status": rng.choice(self.statuses),

            "is_fraud": bool(fraud),

            "fraud_scenario": scenario,

            "occurred_at": occurred_at

        }

        self.records.append(record)

        self.transaction_id += 1

    # ======================================
    # NORMAL TRANSACTIONS
    # ======================================

    def generate_normal_transactions(self):

        print("\nGenerating normal transactions...")

        for _ in range(TOTAL_NORMAL):

            account = self.random_account()

            baseline = self.account_baseline[account]

            amount = rng.lognormal(
                mean=np.log(baseline),
                sigma=0.55
            )

            amount = min(amount, 10000)

            transaction_type = rng.choice(
                self.transaction_types
            )

            self.create_transaction(
                account=account,
                amount=amount,
                occurred_at=self.random_time(),
                transaction_type=transaction_type,
                fraud=False,
                scenario="normal"
            )

        print(f"Normal transactions: {TOTAL_NORMAL}")

    # ======================================
    # FRAUD SCENARIO 1: HIGH VALUE DEVIATION
    # ======================================

    def generate_high_value_fraud(self):

        print("\nGenerating high-value fraud...")

        for _ in range(200):

            account = self.random_account()

            baseline = self.account_baseline[account]

            amount = baseline * rng.uniform(15, 25)

            self.create_transaction(
                account=account,
                amount=amount,
                occurred_at=self.random_time(),
                transaction_type="purchase",
                fraud=True,
                scenario="high_value_deviation"
            )

    # ======================================
    # FRAUD SCENARIO 2: NIGHT-TIME ANOMALY
    # ======================================

    def generate_night_fraud(self):

        print("\nGenerating night-time fraud...")

        for _ in range(200):

            account = self.random_account()

            baseline = self.account_baseline[account]

            day = int(rng.integers(0, 90))

            hour = int(rng.integers(0, 5))

            minute = int(rng.integers(0, 60))

            occurred_at = (
                self.start_time +
                timedelta(
                    days=day,
                    hours=hour,
                    minutes=minute
                )
            )

            amount = baseline * rng.uniform(3, 5)

            self.create_transaction(
                account=account,
                amount=amount,
                occurred_at=occurred_at,
                transaction_type="purchase",
                fraud=True,
                scenario="night_time_anomaly"
            )

    # ======================================
    # FRAUD SCENARIO 3: RAPID TRANSFERS
    # ======================================

    def generate_rapid_transfer_fraud(self):

        print("\nGenerating rapid-transfer fraud...")

        for _ in range(40):

            account = self.random_account()

            baseline = self.account_baseline[account]

            base_time = self.random_time()

            for i in range(5):

                occurred_at = (
                    base_time +
                    timedelta(minutes=i * 2)
                )

                amount = baseline * rng.uniform(1.5, 3)

                self.create_transaction(
                    account=account,
                    amount=amount,
                    occurred_at=occurred_at,
                    transaction_type="transfer",
                    channel="upi",
                    fraud=True,
                    scenario="rapid_transfer_burst"
                )

    # ======================================
    # FRAUD SCENARIO 4: SPENDING DEVIATION
    # ======================================

    def generate_spending_deviation_fraud(self):

        print("\nGenerating spending-deviation fraud...")

        for _ in range(200):

            account = self.random_account()

            baseline = self.account_baseline[account]

            amount = baseline * rng.uniform(8, 12)

            self.create_transaction(
                account=account,
                amount=amount,
                occurred_at=self.random_time(),
                transaction_type="purchase",
                fraud=True,
                scenario="spending_deviation"
            )

    # ======================================
    # FRAUD SCENARIO 5: MERCHANT BURST
    # ======================================

    def generate_merchant_burst_fraud(self):

        print("\nGenerating merchant-burst fraud...")

        for _ in range(50):

            account = self.random_account()

            baseline = self.account_baseline[account]

            merchant = self.random_merchant()

            base_time = self.random_time()

            for i in range(4):

                occurred_at = (
                    base_time +
                    timedelta(minutes=i * 3)
                )

                amount = baseline * rng.uniform(1.5, 2.5)

                self.create_transaction(
                    account=account,
                    amount=amount,
                    occurred_at=occurred_at,
                    transaction_type="purchase",
                    fraud=True,
                    scenario="merchant_burst",
                    merchant_id=merchant
                )

    # ======================================
    # BUILD DATASET
    # ======================================

    def generate(self):

        self.generate_normal_transactions()

        self.generate_high_value_fraud()

        self.generate_night_fraud()

        self.generate_rapid_transfer_fraud()

        self.generate_spending_deviation_fraud()

        self.generate_merchant_burst_fraud()

        df = pd.DataFrame(self.records)

        df = df.sort_values(
            "occurred_at"
        ).reset_index(drop=True)

        df["transaction_id"] = np.arange(
            1,
            len(df) + 1
        )

        return df


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n===================================")
    print("SYNTHETIC FRAUD GENERATOR V2")
    print("===================================")

    accounts, merchants = load_reference_data()

    generator = FraudDatasetGenerator(
        accounts,
        merchants
    )

    df = generator.generate()

    # Validate dataset
    assert len(df) == TOTAL_TRANSACTIONS

    assert df["is_fraud"].sum() == TOTAL_FRAUD

    assert not df["amount"].le(0).any()

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Export CSV
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===================================")
    print("DATASET GENERATION SUMMARY")
    print("===================================")

    print(f"Total transactions: {len(df)}")

    print(f"Normal transactions: {(~df['is_fraud']).sum()}")

    print(f"Fraud transactions: {df['is_fraud'].sum()}")

    print(
        f"Fraud percentage: "
        f"{df['is_fraud'].mean() * 100:.2f}%"
    )

    print("\nFraud Scenario Distribution:")

    print(
        df[df["is_fraud"]]["fraud_scenario"]
        .value_counts()
    )

    print("\nDataset Preview:")

    print(df.head(10).to_string(index=False))

    print("\n===================================")

    print("DATASET SAVED SUCCESSFULLY")

    print("===================================")

    print(f"Output: {OUTPUT_PATH}")

    print("\nOriginal PostgreSQL database was not modified.")


if __name__ == "__main__":

    main()

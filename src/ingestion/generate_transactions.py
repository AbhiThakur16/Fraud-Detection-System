import random
import uuid

from datetime import datetime, timedelta, timezone

from psycopg2.extras import execute_values

from src.db_connection import get_connection


# ==========================================
# CONFIGURATION
# ==========================================

TOTAL_USERS = 1000
TOTAL_ACCOUNTS = 1500
TOTAL_MERCHANTS = 100
TOTAL_TRANSACTIONS = 50000

BATCH_SIZE = 1000

random.seed(42)


# ==========================================
# SYNTHETIC DATA
# ==========================================

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun",
    "Rahul", "Rohan", "Abhishek", "Karan",
    "Priya", "Ananya", "Neha", "Aditi",
    "Sneha", "Kavya", "Isha", "Meera"
]

LAST_NAMES = [
    "Sharma", "Verma", "Singh", "Kumar",
    "Gupta", "Thakur", "Rao", "Malhotra",
    "Patel", "Mehta"
]

CATEGORIES = [
    "grocery", "electronics", "fashion",
    "travel", "food", "healthcare",
    "entertainment", "fuel", "online_services"
]

TRANSACTION_TYPES = [
    "purchase", "transfer", "withdrawal",
    "deposit", "refund"
]

CHANNELS = [
    "card", "upi", "net_banking",
    "atm", "mobile", "web"
]

STATUSES = [
    "settled", "settled", "settled",
    "authorized", "declined"
]


# ==========================================
# GENERATE USERS
# ==========================================

def generate_users(cursor):

    users = []

    for i in range(TOTAL_USERS):

        name = (
            f"{random.choice(FIRST_NAMES)} "
            f"{random.choice(LAST_NAMES)}"
        )

        users.append((
            name,
            f"customer{i+1}@example.com",
            f"9{random.randint(100000000, 999999999)}",
            "IN"
        ))

    query = """
        INSERT INTO users
        (full_name, email, phone, country_code)
        VALUES %s
        RETURNING user_id;
    """

    result = execute_values(
        cursor,
        query,
        users,
        page_size=BATCH_SIZE,
        fetch=True
    )

    return [row[0] for row in result]


# ==========================================
# GENERATE ACCOUNTS
# ==========================================

def generate_accounts(cursor, user_ids):

    accounts = []

    for i in range(TOTAL_ACCOUNTS):

        accounts.append((
            random.choice(user_ids),
            f"IN{1000000000000000000000 + i}",
            random.choice([
                "savings", "current", "credit"
            ]),
            "INR",
            "active"
        ))

    query = """
        INSERT INTO accounts
        (
            user_id,
            account_number,
            account_type,
            currency,
            account_status
        )
        VALUES %s
        RETURNING account_id;
    """

    result = execute_values(
        cursor,
        query,
        accounts,
        page_size=BATCH_SIZE,
        fetch=True
    )

    return [row[0] for row in result]


# ==========================================
# GENERATE MERCHANTS
# ==========================================

def generate_merchants(cursor):

    merchants = []

    for i in range(TOTAL_MERCHANTS):

        merchants.append((
            f"Merchant {i+1}",
            f"MER-{i+1:04d}",
            random.choice(CATEGORIES),
            "IN",
            True
        ))

    query = """
        INSERT INTO merchants
        (
            merchant_name,
            merchant_code,
            merchant_category,
            country_code,
            is_active
        )
        VALUES %s
        RETURNING merchant_id;
    """

    result = execute_values(
        cursor,
        query,
        merchants,
        page_size=BATCH_SIZE,
        fetch=True
    )

    return [row[0] for row in result]


# ==========================================
# GENERATE TRANSACTIONS
# ==========================================

def generate_transactions(cursor, account_ids, merchant_ids):

    total_fraud = 0

    now = datetime.now(timezone.utc)

    query = """
        INSERT INTO transactions
        (
            external_reference,
            sender_account_id,
            receiver_account_id,
            merchant_id,
            transaction_type,
            transaction_channel,
            amount,
            currency,
            transaction_status,
            is_fraud,
            occurred_at
        )
        VALUES %s;
    """

    for start in range(0, TOTAL_TRANSACTIONS, BATCH_SIZE):

        batch = []

        current_size = min(
            BATCH_SIZE,
            TOTAL_TRANSACTIONS - start
        )

        for _ in range(current_size):

            sender = random.choice(account_ids)

            transaction_type = random.choice(
                TRANSACTION_TYPES
            )

            receiver = None
            merchant = None

            if transaction_type == "transfer":

                receiver = random.choice(account_ids)

                while receiver == sender:
                    receiver = random.choice(account_ids)

            elif transaction_type in ["purchase", "refund"]:

                merchant = random.choice(merchant_ids)

            amount = round(
                random.lognormvariate(6, 1.1),
                2
            )

            amount = min(amount, 500000)

            # Synthetic fraud labels for initial experiments
            is_fraud = (
                amount > 100000
                or random.random() < 0.008
            )

            if is_fraud:
                total_fraud += 1

            occurred_at = now - timedelta(
                seconds=random.randint(0, 7776000)
            )

            batch.append((
                str(uuid.uuid4()),
                sender,
                receiver,
                merchant,
                transaction_type,
                random.choice(CHANNELS),
                amount,
                "INR",
                random.choice(STATUSES),
                is_fraud,
                occurred_at
            ))

        execute_values(
            cursor,
            query,
            batch,
            page_size=BATCH_SIZE
        )

        print(
            f"Inserted {start + current_size:,} "
            f"/ {TOTAL_TRANSACTIONS:,} transactions"
        )

    return total_fraud


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        # Prevent accidental duplicate generation
        cursor.execute("""
            SELECT
                (SELECT COUNT(*) FROM users),
                (SELECT COUNT(*) FROM accounts),
                (SELECT COUNT(*) FROM merchants),
                (SELECT COUNT(*) FROM transactions);
        """)

        counts = cursor.fetchone()

        if any(count > 0 for count in counts):

            raise RuntimeError(
                "Database already contains records. "
                "Generation stopped to prevent duplicates."
            )

        print("\nStarting synthetic data generation...\n")

        user_ids = generate_users(cursor)

        print(f"Users created: {len(user_ids):,}")

        account_ids = generate_accounts(
            cursor, user_ids
        )

        print(f"Accounts created: {len(account_ids):,}")

        merchant_ids = generate_merchants(cursor)

        print(f"Merchants created: {len(merchant_ids):,}")

        fraud_count = generate_transactions(
            cursor,
            account_ids,
            merchant_ids
        )

        connection.commit()

        print("\n================================")
        print("DATA GENERATION SUCCESSFUL")
        print("================================")

        print(f"Total users: {len(user_ids):,}")
        print(f"Total accounts: {len(account_ids):,}")
        print(f"Total merchants: {len(merchant_ids):,}")
        print(f"Total transactions: {TOTAL_TRANSACTIONS:,}")
        print(f"Synthetic fraud labels: {fraud_count:,}")

    except Exception:

        connection.rollback()
        raise

    finally:

        connection.close()


if __name__ == "__main__":
    main()
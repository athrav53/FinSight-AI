
import pandas as pd
from db.connection import get_connection


# Run SQL query and return results
def run_query(query, params=None):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


# Get all available users
def get_users():
    return run_query("""
        SELECT DISTINCT user_id
        FROM bank_transactions
        WHERE user_id IS NOT NULL
        ORDER BY user_id
    """)


# Get earliest and latest transaction dates
def get_date_range():
    result = run_query("""
        SELECT
            MIN(transaction_date),
            MAX(transaction_date)
        FROM bank_transactions
    """)

    return result[0]


# Fetch transactions using optional filters
def get_transactions(user_id=None, start_date=None, end_date=None):

    query = """
        SELECT
            transaction_date,
            user_id,
            merchant,
            category,
            transaction_type,
            amount,
            balance,
            description
        FROM bank_transactions
        WHERE 1=1
    """

    params = []

    # Filter by user only when selected
    if user_id is not None:
        query += " AND user_id = %s"
        params.append(user_id)

    # Filter by start date
    if start_date is not None:
        query += " AND transaction_date >= %s"
        params.append(start_date)

    # Filter by end date
    if end_date is not None:
        query += " AND transaction_date <= %s"
        params.append(end_date)

    query += " ORDER BY transaction_date DESC, id DESC"

    return run_query(query, tuple(params))


# Convert SQL results into Pandas DataFrame
def get_dashboard_data(user_id=None, start_date=None, end_date=None):

    rows = get_transactions(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

    columns = [
        "Date",
        "User ID",
        "Merchant",
        "Category",
        "Type",
        "Amount",
        "Balance",
        "Description"
    ]

    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Amount"] = pd.to_numeric(df["Amount"])
        df["Balance"] = pd.to_numeric(df["Balance"])

    return df


# Test database analytics directly
if __name__ == "__main__":

    print("\n===== FINSIGHT AI ANALYTICS =====")

    print("\nAvailable Users:")
    print(get_users())

    print("\nTransaction Date Range:")
    print(get_date_range())

    print("\nTotal Transactions:")
    df = get_dashboard_data()
    print(len(df))

    print("\nFirst 5 Transactions:")
    print(df.head())

    print("\nTotal Spending:")
    if not df.empty:
        spending = df[df["Type"] == "Debit"]["Amount"].sum()
        print(spending)

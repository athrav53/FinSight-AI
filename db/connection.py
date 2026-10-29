
"""
FinSight AI - PostgreSQL Database Connection

Features:
- PostgreSQL connection using environment variables
- Connection timeout
- Safe error handling
- Database connectivity testing
"""

import os
import logging

import psycopg
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def get_connection():
    """
    Create and return a PostgreSQL connection.

    Raises:
        ConnectionError: When database connectivity fails.
    """

    try:
        connection = psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "finsight_ai"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10
        )

        return connection

    except psycopg.Error as error:

        logger.error(
            "PostgreSQL connection failed: %s",
            error
        )

        raise ConnectionError(
            "Unable to connect to the FinSight AI database. "
            "Please verify PostgreSQL is running and "
            "your database configuration is correct."
        ) from error


def test_connection():
    """
    Test PostgreSQL connectivity.

    Returns:
        True if connection succeeds, otherwise False.
    """

    try:

        with get_connection() as conn:

            with conn.cursor() as cursor:

                cursor.execute(
                    "SELECT current_database(), version()"
                )

                database, version = cursor.fetchone()

                print("PostgreSQL connected successfully!")
                print("Database:", database)
                print("Version:", version.split(",")[0])

                return True

    except (ConnectionError, psycopg.Error) as error:

        print("Database connection failed!")
        print("Error:", error)

        return False


if __name__ == "__main__":
    test_connection()
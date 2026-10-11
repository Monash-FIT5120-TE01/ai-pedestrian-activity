import os
import psycopg2


def get_db_connection(db_url):
    """Connect using standard PostgreSQL environment variables."""
    return psycopg2.connect(db_url)
    
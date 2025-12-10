import mysql.connector
from mysql.connector import pooling
from typing import Any, List, Optional

# Create connection pool
pool = pooling.MySQLConnectionPool(
    pool_name="expression_besoin_pool",
    pool_size=10,
    host='localhost',
    user='root',
    password='',
    database='expression_besoin'
)

async def query(text: str, params: Optional[List[Any]] = None):
    """Execute a SQL query and return the results"""
    connection = pool.get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(text, params or [])
        
        # Check if it's a SELECT query
        if text.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            return rows
        else:
            # For INSERT, UPDATE, DELETE
            connection.commit()
            return {"affected_rows": cursor.rowcount, "last_insert_id": cursor.lastrowid}
    finally:
        cursor.close()
        connection.close()
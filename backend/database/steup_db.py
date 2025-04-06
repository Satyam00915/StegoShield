import psycopg2
from db_config import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    with open("backend/database/schema.sql", "r") as f:
        schema = f.read()
        cursor.execute(schema)  # Use execute only if schema has one statement, otherwise use executescript alternative

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ PostgreSQL Tables created successfully.")

if __name__ == "__main__":
    create_tables()

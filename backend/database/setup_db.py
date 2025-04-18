import psycopg2
from db_config import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    try:
        with open("backend/database/schema.sql", "r") as f:
            schema = f.read()
            print("🛠️ Executing full schema.sql content")
            cursor.execute(schema)  # 🚀 Don't split by ';' anymore

        conn.commit()
        print("✅ PostgreSQL Tables created successfully.")

    except Exception as e:
        print("❌ Error executing schema.sql:", e)

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_tables()

import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="stegoshield",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
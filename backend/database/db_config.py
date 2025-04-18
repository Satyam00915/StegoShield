import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    db_url = ('DATABASE_URL') 
    return psycopg2.connect(db_url)

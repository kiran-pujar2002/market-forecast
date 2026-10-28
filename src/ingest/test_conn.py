import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)

with engine.connect() as conn:
    version = conn.execute(text("SELECT version();")).scalar()
    print("Connected:", version)
import os, sqlite3
from typing import List, Tuple
import pandas as pd
from app.models import Product, Review

DB_PATH = os.getenv("APP_DB_PATH", "data/app.db")

def ensure_db():
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS products(
            product_id TEXT PRIMARY KEY,
            title TEXT, category TEXT, price REAL, description TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS reviews(
            review_id TEXT PRIMARY KEY,
            product_id TEXT, rating REAL, text TEXT,
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        )""")
        conn.commit()

def upsert_products(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("products", conn, if_exists="append", index=False)
        # deduplicate by product_id
        conn.execute("""DELETE FROM products
                        WHERE rowid NOT IN (
                          SELECT MIN(rowid) FROM products GROUP BY product_id
                        )""")
        conn.commit()

def upsert_reviews(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("reviews", conn, if_exists="append", index=False)
        conn.execute("""DELETE FROM reviews
                        WHERE rowid NOT IN (
                          SELECT MIN(rowid) FROM reviews GROUP BY review_id
                        )""")
        conn.commit()

def get_all_products() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM products", conn)
    # BEFORE (buggy): tried to use r._asdict() / row together
    # return [Product(**r._asdict() if hasattr(r,"_asdict") else r._asdict())
    #         if hasattr(r,"_asdict") else Product(**row) for row in df.to_dict(orient="records")]

    # AFTER (correct):
    return [Product(**row) for row in df.to_dict(orient="records")]


def get_product(pid: str) -> Product | None:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM products WHERE product_id = ?", conn, params=[pid])
    recs = df.to_dict(orient="records")
    return Product(**recs[0]) if recs else None

def get_reviews(pid: str) -> List[Review]:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM reviews WHERE product_id = ?", conn, params=[pid])
    return [Review(**row) for row in df.to_dict(orient="records")]

def load_csv(products_csv: str, reviews_csv: str):
    ensure_db()
    p = pd.read_csv(products_csv, dtype=str)
    # best-effort typing
    if "price" in p.columns:
        p["price"] = pd.to_numeric(p["price"], errors="coerce")
    upsert_products(p[["product_id","title","category","price","description"]].fillna(""))
    r = pd.read_csv(reviews_csv, dtype=str)
    if "rating" in r.columns:
        r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    upsert_reviews(r[["review_id","product_id","rating","text"]].fillna(""))

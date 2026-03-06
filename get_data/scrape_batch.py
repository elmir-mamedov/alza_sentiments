"""scrape_batch.py
Scrape reviews for a batch of products from url_queue and save to reviews.csv
"""
import csv
import time
import random
import re
from curl_cffi.requests import Session
from db import get_connection

BATCH_SIZE = 5
REVIEWS_FILE = "reviews.csv"
PRODUCTS_FILE = "product_stats.csv"

def get_tracking_params(session) -> tuple[str, str]:
    r = session.get('https://www.alza.cz', impersonate='chrome120')
    ucik  = re.search(r'ucik=(u_[a-z0-9_]+)', r.text)
    pgrik = re.search(r'pgrik=(p_[a-z0-9_]+)', r.text)
    return (
        ucik.group(1)  if ucik  else "",
        pgrik.group(1) if pgrik else "",
    )

def fetch_product_stats(session, commodity_id: int, ucik: str, pgrik: str) -> dict:
    url = (
        f"https://webapi.alza.cz/api/catalog/v2/commodities/"
        f"{commodity_id}/reviewStats?country=CZ&ucik={ucik}&pgrik={pgrik}"
    )
    try:
        response = session.get(url, impersonate="chrome120", timeout=10)
        if not response.text.strip():
            return {}
        data = response.json()
        return {
            "commodity_id":          commodity_id,
            "name":                  data.get("name", ""),
            "rating_average":        data.get("ratingAverage"),
            "rating_count":          data.get("ratingCount"),
            "purchase_count":        data.get("purchaseCountFormatted", ""),
        }
    except Exception as e:
        print(f"  ERROR fetching stats: {e}")
        return {}


def fetch_reviews(session, commodity_id: int) -> list[dict]:
    reviews = []
    limit = 100
    offset = 0
    max_retries = 3

    while True:
        url = (
            f"https://webapi.alza.cz/api/catalog/v2/commodities/"
            f"{commodity_id}/reviews?country=cz&limit={limit}&offset={offset}"
        )
        for attempt in range(max_retries):
            try:
                response = session.get(url, impersonate="chrome120", timeout=10)
                if not response.text.strip():
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt + random.uniform(1, 3)
                        print(f"  Empty response, retrying in {wait:.1f}s...")
                        time.sleep(wait)
                        continue
                    else:
                        return reviews
                data = response.json()
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                return reviews

        for review in data.get("value", []):
            for text in review.get("positives", []):
                if text.strip():
                    reviews.append({"text": text.strip(), "label": 1, "commodity_id": commodity_id})
            for text in review.get("negatives", []):
                if text.strip():
                    reviews.append({"text": text.strip(), "label": 0, "commodity_id": commodity_id})

        offset += limit
        if offset >= data.get("paging", {}).get("size", 0):
            break

    return reviews


def append_to_csv(filepath, rows, fieldnames):
    write_header = False
    try:
        with open(filepath, "r") as f:
            write_header = f.readline() == ""
    except FileNotFoundError:
        write_header = True

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    all_reviews = []
    all_stats = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT url, commodity_id FROM url_queue
                WHERE processed = false
                  AND commodity_id IS NOT NULL
                LIMIT %s;
                """,
                (BATCH_SIZE,),
            )
            rows = cur.fetchall()

    print(f"Fetched {len(rows)} URLs to process")

    with Session() as session:
        ucik, pgrik = get_tracking_params(session)
        print(f"Got ucik: {ucik} | pgrik: {pgrik}")

        for i, row in enumerate(rows, 1):
            url = row["url"]
            commodity_id = row["commodity_id"]
            print(f"[{i}/{len(rows)}] Scraping commodity {commodity_id} — {url}")

            # Fetch product stats
            stats = fetch_product_stats(session, commodity_id, ucik, pgrik)
            if stats:
                all_stats.append(stats)
                print(f"  → {stats.get('name', '?')} | rating: {stats.get('rating_average')} ({stats.get('rating_count')} ratings)")

            # Fetch reviews
            reviews = fetch_reviews(session, commodity_id)
            print(f"  → {len(reviews)} reviews")
            all_reviews.extend(reviews)

            # Mark as processed
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE url_queue SET processed = true WHERE url = %s;",
                        (url,),
                    )
                conn.commit()

            time.sleep(random.uniform(4.5, 8.0))

            if random.random() < 0.15:
                pause = random.uniform(8.0, 20.0)
                print(f"  [human pause: {pause:.1f}s]")
                time.sleep(pause)

    append_to_csv(
        REVIEWS_FILE,
        all_reviews,
        fieldnames=["commodity_id", "text", "label"]
    )
    append_to_csv(
        PRODUCTS_FILE,
        all_stats,
        fieldnames=["commodity_id", "name", "rating_average", "rating_count", "purchase_count"]
    )

    print(f"\nDone. {len(all_reviews)} reviews → {REVIEWS_FILE}")
    print(f"      {len(all_stats)} products → {PRODUCTS_FILE}")

    import pandas as pd

    reviews_df = pd.read_csv(REVIEWS_FILE)
    stats_df = pd.read_csv(PRODUCTS_FILE)

    merged = reviews_df.merge(stats_df, on="commodity_id", how="left")
    merged.to_csv("merged.csv", index=False)

    print(f"      {len(merged)} rows → merged.csv")


if __name__ == "__main__":
    main()
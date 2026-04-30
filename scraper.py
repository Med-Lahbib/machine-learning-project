"""
Enhanced web scraper for books.toscrape.com.

Scrapes all books across all pages, including visiting each individual
book page to extract the full description. Demonstrates real data
sourcing depth: pagination + detail page scraping.

Usage:
    python scraper.py
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm

BASE_URL = "http://books.toscrape.com/catalogue/"
PAGE_URL = BASE_URL + "page-{}.html"


def fetch_with_retry(url, max_retries=3):
    """Fetch a URL with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            return response
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return None
    return None


def get_book_description(book_relative_url):
    """Visit an individual book page and extract the full description."""
    url = BASE_URL + book_relative_url
    try:
        response = fetch_with_retry(url)
        if response is None or response.status_code != 200:
            return "No description available."

        soup = BeautifulSoup(response.text, "html.parser")

        # The description is in a <p> tag after the <div id="product_description">
        desc_header = soup.find("div", id="product_description")
        if desc_header:
            description = desc_header.find_next_sibling("p")
            if description:
                return description.text.strip()

        return "No description available."
    except Exception as e:
        return "No description available."


def scrape_all_books():
    """Scrape all books from books.toscrape.com with full descriptions."""

    all_books = []
    page_number = 1

    print("=" * 60)
    print("Books.toscrape.com - Enhanced Scraper")
    print("=" * 60)

    # Phase 1: Collect all book listing data
    print("\n[Phase 1] Collecting book listings...")
    while True:
        url = PAGE_URL.format(page_number)
        response = fetch_with_retry(url)

        if response is None or response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.find_all("article", class_="product_pod")

        if not books:
            break

        for book in books:
            title = book.find("h3").find("a")["title"]
            price = book.find("p", class_="price_color").text
            price = float(price[2:])  # Remove currency symbol '£'
            rating = book.find("p", class_="star-rating")["class"][1]
            book_url = book.find("h3").find("a")["href"]
            all_books.append({
                "title": title,
                "price": price,
                "rating": rating,
                "url": book_url,
            })

        print(f"  Page {page_number}: {len(books)} books found")
        page_number += 1

    print(f"\n  Total books collected: {len(all_books)}")

    # Phase 2: Fetch individual book descriptions
    print("\n[Phase 2] Fetching book descriptions (this may take a few minutes)...")
    for book in tqdm(all_books, desc="  Scraping descriptions"):
        book["description"] = get_book_description(book["url"])

    # Count books with descriptions
    with_desc = sum(1 for b in all_books if b["description"] != "No description available.")
    print(f"\n  Books with descriptions: {with_desc}/{len(all_books)}")

    # Phase 3: Save to CSV
    print("\n[Phase 3] Saving to books.csv...")
    with open("books.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Price", "Rating", "Description", "URL"])
        for book in all_books:
            writer.writerow([
                book["title"],
                book["price"],
                book["rating"],
                book["description"],
                book["url"],
            ])

    print(f"  ✓ Saved {len(all_books)} books to books.csv")
    print("\n" + "=" * 60)
    print("Scraping complete!")
    print("=" * 60)


if __name__ == "__main__":
    scrape_all_books()
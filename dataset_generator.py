"""
    Dataset Generation and Preparation Pipeline (V3 - Official API)
    ----------------------------------------------------------------

    This script automates gathering image data by making direct requests to the
    official Brave Search Image API. This is the most reliable method.

    Part 1: Image Scraping (This Script)
    1.  Defines a list of GENERAL categories (e.g., 'pastry', 'cake').
    2.  For each category, it generates several diverse search queries for realistic scenes.
    3.  A dedicated function calls the Brave Image Search API endpoint with the
        correct headers (including the API key) and parameters.
    4.  The JSON response is parsed to extract direct image URLs.
    5.  The images are downloaded and saved into organized folders.

    Part 2: Data Annotation and Processing (Roboflow)
    (The process remains the same: clean, upload, annotate, generate, and export.)
"""

import os
import requests
import time
import logging
import certifi
from serpapi import GoogleSearch

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{')

# Your API Key
BRAVE_API_KEY = "BSA_nezW90Xg_ElKdeJGVbeJJqMCEl_"
SERPAPI_API_KEY = "155b6ce03e7e63b825e39edc7108dd6d44190da78fe4fc8efa60fce6bb14fda2"

# UPDATED: Differentiated image counts based on category complexity
IMAGE_TARGETS = {
    "pastry": 190,
    "bread": 125,
    "cake": 125,
    "donut": 60,
    "drink": 60,
}

BASE_DOWNLOAD_PATH = "bakery_serpapi_dataset"

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}

# --- Functions ---


# NEW: Function to call the SerpApi Google Images API
def search_serpapi_images(query, target_count):
    """
    Searches for images using SerpApi, handling pagination to fetch a target number of results.
    """
    all_urls = []
    page_num = 0

    while len(all_urls) < target_count:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "ijn": page_num,  # 'ijn' is the page number parameter
            "tbm": "isch",  # 'isch' specifies image search
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            image_results = results.get("images_results", [])

            if not image_results:
                # Stop if the API returns no more results
                break

            # Extract the 'original' image URL from each result
            urls = [res['original'] for res in image_results if 'original' in res]
            all_urls.extend(urls)

            page_num += 1  # Move to the next page for the next request
            time.sleep(1)  # Be respectful between paginated requests

        except Exception as e:
            logging.error(f"SerpApi request failed for query '{query}' on page {page_num}: {e}")
            break

    return all_urls

# NEW: Function to call the official Brave API
def search_brave_images(query, count):
    """Searches for images using the official Brave Image Search API."""
    api_url = "https://api.search.brave.com/res/v1/images/search"

    params = {
        "q": query,
        "count": count,

    }

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": BRAVE_API_KEY
    }

    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=20, verify=certifi.where())
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        # The image URL is in the 'src' key of each result
        return [res['properties']['url'] for res in data.get('results', []) if 'properties' in res and 'url' in res['properties']]

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for query '{query}': {e}")
        return []


def generate_search_queries(category):
    """Generates a list of diverse search queries for a given category."""
    if category == "beverage":
        return [
            "food photography cafe shop with drinks juices display"
        ]

    # General queries for baked goods
    return [
        f"food photography bakery {category} display",
    ]


def download_images(urls, output_folder, category_name):
    """Downloads images from a list of URLs into a specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created directory: {output_folder}")

    downloaded_count = 0
    for i, url in enumerate(set(urls)):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            response.raise_for_status()
            file_path = os.path.join(output_folder, f"{category_name}_{i + 1}.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            downloaded_count += 1
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            logging.error(f"An unexpected error occurred for URL {url}: {e}")

    return downloaded_count

# --- Main Execution ---


if __name__ == "__main__":
    if not BRAVE_API_KEY or "YOUR_BRAVE_API_KEY" in BRAVE_API_KEY:
        logging.error("Brave API key is not set. Please edit the script and add your key.")
    else:
        if not os.path.exists(BASE_DOWNLOAD_PATH):
            os.makedirs(BASE_DOWNLOAD_PATH)

        for category, target_count in IMAGE_TARGETS.items():
            category_folder_name = category.replace(" ", "_")
            output_directory = os.path.join(BASE_DOWNLOAD_PATH, category_folder_name)
            logging.info(f"--- Processing category: {category.upper()} (Target: {target_count} images) ---")

            search_queries = generate_search_queries(category)
            all_category_urls = []

            for query in search_queries:
                logging.info(f"  -> Searching for: '{query}'")
                # MODIFIED: Call our new API function
                # urls = search_brave_images(query, count=target_count)
                urls = search_serpapi_images(query, target_count=target_count)
                all_category_urls.extend(urls)
                logging.info(f"     Found {len(urls)} URLs for this query.")

                # 1 request per second with the free subscription
                time.sleep(1)

            if all_category_urls:
                unique_urls = list(set(all_category_urls))
                logging.info(f"Found a total of {len(unique_urls)} unique URLs for '{category}'. Starting download...")
                downloaded_count = download_images(unique_urls, output_directory, category_folder_name)
                logging.info(f"Successfully downloaded {downloaded_count} images for '{category}'.")
            else:
                logging.warning(f"No URLs found for the entire category: {category}")

        logging.info("--- All categories processed. Download complete! ---")

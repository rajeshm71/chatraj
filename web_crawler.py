import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def get_all_pages(root_url, max_pages=100):
    visited = set()  # To track visited URLs
    to_visit = [root_url]  # Initialize with the root URL
    all_pages = []

    while to_visit and len(all_pages) < max_pages:
        current_url = to_visit.pop(0)

        # Skip if already visited
        if current_url in visited:
            continue

        try:
            # Fetch the page content
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            visited.add(current_url)
            all_pages.append(current_url)
            print(f"Visited: {current_url}")

            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)

                # Filter out non-HTTP links and avoid external links
                if full_url.startswith(root_url) and full_url not in visited:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Error visiting {current_url}: {e}")
        time.sleep(1)  # Add delay to avoid overwhelming the server

    return all_pages

# Example usage
import urllib.robotparser

rp = urllib.robotparser.RobotFileParser()
root_url = "https://www.intuitionmind.ai"
rp.set_url( "https://www.intuitionmind.ai/robots.txt")
rp.read()
if rp.can_fetch("*", root_url):
    print("Allowed to crawl")
else:
    print("Crawling disallowed")

pages = get_all_pages(root_url)
print(f"Discovered {len(pages)} pages:")
for page in pages:
    print(page)

import requests
from bs4 import BeautifulSoup
import time
from db import conditions_collection

BASE_URL = "https://www.nhsinform.scot"
AZ_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def clean(text):
    return " ".join(text.split())


def get_links():
    response = requests.get(AZ_URL, headers=HEADERS, timeout=15)

    print("A-Z Status:", response.status_code)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/illnesses-and-conditions/" in href and "a-to-z" not in href:
            if href.startswith("http"):
                full_url = href
            else:
                full_url = BASE_URL + href

            full_url = full_url.split("#")[0].rstrip("/")

            if full_url not in links:
                links.append(full_url)

    print("Total links found:", len(links))
    return links


def get_section(soup, keywords):
    text = ""

    for tag in soup.find_all(["h2", "h3"]):
        title = tag.get_text(" ", strip=True).lower()

        if any(keyword in title for keyword in keywords):
            sibling = tag.find_next_sibling()

            while sibling and sibling.name not in ["h2", "h3"]:
                if sibling.name in ["p", "ul", "ol"]:
                    text += " " + clean(sibling.get_text(" ", strip=True))

                sibling = sibling.find_next_sibling()

    return text.strip()


def scrape_condition(url):
    try:
        print("Scraping:", url)

        response = requests.get(url, headers=HEADERS, timeout=15)
        print("Page status:", response.status_code)

        if response.status_code != 200:
            print("Skipped because status is:", response.status_code)
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h1")

        if not title_tag:
            print("Skipped because no title found")
            return None

        title = clean(title_tag.get_text(" ", strip=True))

        if not title or title.lower() in ["unknown", "illnesses and conditions"]:
            print("Skipped invalid title:", title)
            return None

        symptoms = get_section(soup, ["symptom", "sign"])
        causes = get_section(soup, ["cause", "causes"])
        warnings = get_section(soup, ["urgent", "emergency", "call", "danger", "help"])
        recommendations = get_section(soup, ["treatment", "self-help", "what to do", "manage"])

        # fallback لو sections طلعت فاضية
        if not symptoms:
            paragraphs = soup.find_all("p")
            symptoms = clean(" ".join([p.get_text(" ", strip=True) for p in paragraphs[:3]]))

        data = {
            "condition": title,
            "url": url,
            "symptoms": symptoms,
            "causes": causes,
            "warnings": warnings,
            "recommendations": recommendations
        }

        conditions_collection.update_one(
            {"condition": title},
            {"$set": data},
            upsert=True
        )

        print("Inserted/Updated:", title)
        return data

    except Exception as e:
        print("Error scraping page:", e)
        return None


def scrape_all(limit=100):
    links = get_links()

    if not links:
        print("No links found")
        return []

    results = []

    for url in links[:limit]:
        data = scrape_condition(url)

        if data:
            results.append(data)

        time.sleep(1)

    print("Scraping finished. Saved:", len(results))
    return results


if __name__ == "__main__":
    scrape_all(limit=100)
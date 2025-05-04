import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def search_wikipedia(query, limit=500):
    url = "https://en.wikipedia.org/w/api.php"
    results = []
    sroffset = 0

    while len(results) < limit:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": min(50, limit - len(results)),
            "sroffset": sroffset,
            "format": "json"
        }

        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()  
            data = res.json()
        except Exception as e:
            print(f"Error fetching or parsing JSON: {e}")
            print(f"Response text: {res.text[:200]}...")  
            break

        batch = data.get("query", {}).get("search", [])
        if not batch:
            break

        results.extend([
            f"https://en.wikipedia.org/wiki/{r['title'].replace(' ', '_')}"
            for r in batch
        ])
        sroffset += len(batch)
        time.sleep(0.1)  

    return results

# scraping each article
def get_article_text(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        title = soup.select_one("h1").text.strip()

        content_div = soup.select_one("div.mw-parser-output")
        if content_div:
            for tag in content_div.select("table, .mwe-math-element, math"):
                tag.decompose()

            paragraphs = content_div.find_all("p", recursive=True)
            clean_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 40:
                    clean_paragraphs.append(text)
            article_text = "\n\n".join(clean_paragraphs)
        else:
            article_text = "Content not found"
    except Exception as e:
        title = "ERROR"
        article_text = f"Failed to parse {url}: {str(e)}"

    return {"Title": title, "Text": article_text, "URL": url}

def scrape_topic_articles(topic="Lithuanian History", limit=500, output_file="lthist_wiki.csv"):
    urls = search_wikipedia(topic, limit=limit)
    print(f"Found {len(urls)} articles for topic: {topic}")

    data = []
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Downloading: {url}")
        article_data = get_article_text(url)
        data.append(article_data)

        # save progress
        if (i + 1) % 100 == 0:
            pd.DataFrame(data).to_csv(output_file, index=False)

        time.sleep(1) 

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} articles to {output_file}")
    return df

# === Run it ===
if __name__ == "__main__":
    scrape_topic_articles(topic="Lithuanian History", limit=500)

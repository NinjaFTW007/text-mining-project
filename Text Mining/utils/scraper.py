import requests

def fetch_hackernews_titles(limit: int = 30) -> list[str]:
    """
    Fetch the top headlines from HackerNews using their public Firebase API.
    Does not require API keys. Excellent for live text mining demos.
    """
    try:
        # Get top story IDs
        top_ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        resp = requests.get(top_ids_url, timeout=5)
        if resp.status_code != 200:
            return []
        
        story_ids = resp.json()[:limit]
        
        titles = []
        for sid in story_ids:
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
            s_resp = requests.get(story_url, timeout=3)
            if s_resp.status_code == 200:
                data = s_resp.json()
                if data and "title" in data:
                    titles.append(data["title"])
                    
        return titles
    except Exception as e:
        print(f"Scraper error: {e}")
        return []

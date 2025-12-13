from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin

@tool
def get_rendered_html(url: str) -> dict:
    """
    Fetch and return the fully rendered HTML of a webpage.
    """
    print("\nFetching and rendering:", url)
    try:
        with sync_playwright() as p:
            # --- FIX: Add arguments to disable sandbox ---
            browser = p.chromium.launch(
                headless=True, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            page = browser.new_page()

            page.goto(url, wait_until="networkidle")
            content = page.content()

            browser.close()

            # Parse images
            soup = BeautifulSoup(content, "html.parser")
            imgs = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
            
            if len(content) > 300000:
                    print("Warning: HTML too large, truncating...")
                    content = content[:300000] + "... [TRUNCATED DUE TO SIZE]"
            return {
                "html": content,
                "images": imgs,
                "url": url
            }

    except Exception as e:
        # It is helpful to print the error to logs so you can see it in HF
        print(f"DEBUG ERROR in get_rendered_html: {e}") 
        return {"error": f"Error fetching/rendering page: {str(e)}"}
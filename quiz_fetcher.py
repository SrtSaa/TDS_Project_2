"""
Quiz Fetcher Module - Handles browser automation and content extraction
Uses Playwright for JavaScript-rendered quiz pages
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeout, Playwright
import logging
from pathlib import Path
import os
import json
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
PAGE_TIMEOUT = 30000  # 30 seconds
NAVIGATION_TIMEOUT = 45000  # 45 seconds
WAIT_FOR_SELECTOR_TIMEOUT = 10000  # 10 seconds

# Global instances for reuse
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None
_browser_context: Optional[BrowserContext] = None


async def get_browser() -> Browser:
    """Get or create a shared browser instance"""
    global _browser, _playwright
    if _browser is None or not _browser.is_connected():
        if _playwright is None:
            _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu'
            ]
        )
    return _browser


async def get_browser_context() -> BrowserContext:
    """Get or create a browser context"""
    global _browser_context
    browser = await get_browser()
    if _browser_context is None:
        _browser_context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
    return _browser_context


async def close_browser():
    """Close browser and cleanup resources"""
    global _browser, _browser_context, _playwright
    
    try:
        if _browser_context:
            await _browser_context.close()
            _browser_context = None
        
        if _browser:
            await _browser.close()
            _browser = None
        
        if _playwright:
            await _playwright.stop()
            _playwright = None
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# def validate_url(url: str) -> bool:
#     """Validate URL format"""
#     try:
#         result = urlparse(url)
#         return all([result.scheme, result.netloc])
#     except Exception:
#         return False


async def wait_for_content_ready(page: Page) -> bool:
    """
    Wait for JavaScript content to be fully rendered
    Uses multiple strategies to ensure content is ready
    """
    try:
        # Wait for DOMContentLoaded
        await page.wait_for_load_state('domcontentloaded', timeout=PAGE_TIMEOUT)
        
        # Wait for network to be idle
        await page.wait_for_load_state('networkidle', timeout=PAGE_TIMEOUT)
        
        # Additional wait for any dynamic content (atob decoding, etc.)
        await asyncio.sleep(2)
        
        # Wait for multiple rendering cycles (some pages have staged rendering)
        for _ in range(3):
            await page.evaluate('() => new Promise(resolve => setTimeout(resolve, 500))')
        
        # Check if body has any content (be lenient - even 1 char is OK)
        body_content = await page.evaluate('() => document.body ? document.body.innerText.trim().length : 0')
        
        # Also check if there's any HTML content at all
        html_content = await page.evaluate('() => document.body ? document.body.innerHTML.length : 0')
        
        # If we have ANY content (text or HTML), consider it ready
        if body_content > 0 or html_content > 0:
            return True
        
        # Even if no content detected, check for common elements
        # Some pages might have content but in hidden/styled elements
        common_selectors = ['main', '[role="main"]', '.quiz', '.content', '#content', 'article', 'div', 'p']
        for selector in common_selectors:
            try:
                element_count = await page.locator(selector).count()
                if element_count > 0:
                    return True
            except Exception:
                continue
        
        # If we got here, really no content found
        return False
        
    except PlaywrightTimeout:
        # Even on timeout, check if we have some content
        try:
            html_content = await page.evaluate('() => document.body ? document.body.innerHTML.length : 0')
            return html_content > 0
        except:
            return False
    except Exception:
        return False


def extract_question_text(page_text: str, html: str) -> Optional[str]:
    """
    Extract question or instruction text from page content
    Uses multiple adaptive heuristics to find the main question
    """
    # Try to find HTML headings first (most reliable)
    heading_patterns = [
        r'<h1[^>]*>([^<]+)</h1>',
        r'<h2[^>]*>([^<]+)</h2>',
        r'<h3[^>]*>([^<]+)</h3>',
    ]
    
    for pattern in heading_patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        for heading in matches:
            heading = heading.strip()
            # Filter out navigation/header elements
            if len(heading) > 15 and not any(word in heading.lower() for word in ['menu', 'nav', 'header', 'footer', 'login', 'sign']):
                return heading
    
    # Look for question-like patterns (lines ending with ?)
    lines = page_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.endswith('?') and len(line) > 20 and len(line) < 500:
            return line
    
    # Look for emphasized or bold text in HTML
    emphasis_patterns = [
        r'<strong[^>]*>([^<]+)</strong>',
        r'<b[^>]*>([^<]+)</b>',
        r'<em[^>]*>([^<]+)</em>',
    ]
    
    for pattern in emphasis_patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        for text in matches:
            text = text.strip()
            if len(text) > 20 and len(text) < 500:
                return text
    
    # Find the longest substantial paragraph
    paragraphs = [line.strip() for line in lines if line.strip()]
    longest = max(paragraphs, key=len, default='') if paragraphs else ''
    if len(longest) > 30 and not longest.startswith('http'):
        return longest
    
    return None


def extract_download_links(page_url: str, html: str) -> List[Dict[str, str]]:
    """
    Extract all potential download links dynamically
    """
    links = []
    
    # Extract all href attributes
    link_pattern = r'href=["\']([^"\']+)["\']'
    all_urls = re.findall(link_pattern, html, re.IGNORECASE)
    
    # Also check src attributes for resources
    src_pattern = r'src=["\']([^"\']+)["\']'
    all_urls.extend(re.findall(src_pattern, html, re.IGNORECASE))
    
    for url in all_urls:
        url = url.strip()
        
        # Skip empty, anchors, javascript, and template literals
        if not url or url.startswith('#') or url.startswith('javascript:') or url.startswith('data:'):
            continue
        
        # Skip unrendered JavaScript template literals
        if '${' in url or '`' in url:
            continue
        
        # Skip relative paths that don't make sense (contain template syntax)
        if url.startswith('./') or url.startswith('../'):
            if '${' in url or '{' in url:
                continue
        
        # Make absolute URL
        try:
            absolute_url = urljoin(page_url, url)
        except Exception:
            continue
        
        # Validate the URL is properly formed
        try:
            parsed = urlparse(absolute_url)
            # Must have scheme and netloc for remote URLs
            if parsed.scheme in ('http', 'https'):
                if not parsed.netloc:
                    continue
            # For file URLs, check path exists
            elif parsed.scheme == 'file':
                # Skip file URLs that contain template syntax
                if '${' in absolute_url or '{' in absolute_url:
                    continue
        except Exception:
            continue
        
        # Parse URL to get file extension
        path = parsed.path.lower()
        
        # Check if it looks like a file (has extension)
        if '.' in path.split('/')[-1]:
            ext = path.split('.')[-1].split('?')[0]  # Remove query params
            if ext and len(ext) <= 10:  # Reasonable extension length
                links.append({
                    'url': absolute_url,
                    'type': ext.upper()
                })
    
    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link['url'] not in seen:
            seen.add(link['url'])
            unique_links.append(link)
    
    return unique_links


def extract_submit_url(page_url: str, html: str, page_text: str) -> Optional[str]:
    """
    Extract submission URL using adaptive strategies
    Looks for any URL that might be related to submission
    """
    # First: explicit detection from "Submission" section or elements with class 'submit-url'
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Strategy A: Find <code> blocks containing submit urls
        code_tags = soup.find_all('code')
        for code in code_tags:
            text = (code.get_text() or '').strip()
            # Heuristic: submission examples usually show /submit/{level} or similar
            if 'submit' in text.lower():
                # If the code includes a span.submit-url with dynamic replacement
                span = code.find('span', class_='submit-url')
                if span:
                    # The script may replace span textContent with origin + "/submit/"
                    span_text = (span.get_text() or '').strip()
                    # Compose full code text (span + rest of code contents) to capture the trailing level number
                    full_text = code.get_text().strip()
                    # Prefer the full text if it already contains absolute URL
                    if full_text.startswith('http'):
                        return full_text
                    # Otherwise, join with page origin
                    from urllib.parse import urljoin
                    candidate = urljoin(page_url, full_text)
                    return candidate
                # If no span, but code tag still contains absolute submit URL
                if text.startswith('http') and '/submit' in text.lower():
                    return text

        # Strategy B: Find any element with 'submit-url' class directly
        span_submit = soup.select_one('.submit-url')
        if span_submit:
            submit_text = (span_submit.get_text() or '').strip()
            # May be followed by a level number outside the span (e.g., "/submit/" + "12")
            # Try to read the enclosing code tag text for full URL
            parent_code = span_submit.find_parent('code')
            if parent_code:
                full_text = (parent_code.get_text() or '').strip()
                if full_text.startswith('http'):
                    return full_text
                from urllib.parse import urljoin
                return urljoin(page_url, full_text)
            # Fallback: join the span text to origin
            from urllib.parse import urljoin
            return urljoin(page_url, submit_text)
    except Exception:
        # Continue with generic strategies if parsing fails
        pass

    potential_urls = []
    # Strategy 1: Look for form action
    form_pattern = r'<form[^>]+action=["\']([^"\']+)["\']'
    for match in re.finditer(form_pattern, html, re.IGNORECASE):
        url = urljoin(page_url, match.group(1))
        potential_urls.append(('form_action', url))
    
    # Strategy 2: Extract ALL URLs from HTML (code tags, links, scripts)
    url_pattern = r'(?:href|src|action|url|fetch|axios|post|put)[\s=:(["\']+(https?://[^\s"\'<>)]+)'
    for match in re.finditer(url_pattern, html, re.IGNORECASE):
        url = match.group(1).strip()
        potential_urls.append(('html_reference', url))
    
    # Strategy 3: Extract URLs from visible text
    text_url_pattern = r'(https?://[^\s\)<>]+)'
    for match in re.finditer(text_url_pattern, page_text):
        url = match.group(1).strip()
        potential_urls.append(('text_reference', url))
    
    # Strategy 4: Look for URLs in <code> or <pre> tags specifically
    code_pattern = r'<(?:code|pre)[^>]*>([^<]*https?://[^<]+)</(?:code|pre)>'
    for match in re.finditer(code_pattern, html, re.IGNORECASE | re.DOTALL):
        url_match = re.search(r'(https?://[^\s<>]+)', match.group(1))
        if url_match:
            url = url_match.group(1).strip()
            potential_urls.append(('code_tag', url))
    
    # Strategy 5: Look in JavaScript code
    js_url_pattern = r'["\']((https?://[^"\'<>]+/api/[^"\'<>]+))["\']'
    for match in re.finditer(js_url_pattern, html, re.IGNORECASE):
        url = match.group(1).strip()
        potential_urls.append(('javascript', url))
    
    # Filter out obvious non-submission URLs
    blacklist_patterns = [
        r'cloudflare', r'analytics', r'beacon', r'tracking', r'google-analytics', r'gtag',
        r'facebook', r'twitter', r'linkedin', r'\.js$', r'\.css$', r'\.jpg$', r'\.jpeg$', r'\.png$', r'\.gif$', r'\.svg$',
        r'\.woff', r'\.ttf', r'\.eot', r'/static/', r'/assets/', r'/cdn/', r'/fonts/',
        # Explicitly avoid game endpoints for submission
        r'/api/game/start', r'/api/game/move'
    ]
    
    # Score URLs based on context and keywords
    scored_urls = []
    submission_keywords = [
        'submit', 'answer', 'response', 'endpoint', 'webhook', 'quiz', 'check'
    ]
    
    for source, url in potential_urls:
        url_lower = url.lower()
        if any(re.search(pattern, url_lower) for pattern in blacklist_patterns):
            continue
        
        score = 0
        
        # Higher score for URLs mentioned in code tags or explicit 'submit' references
        if source == 'code_tag':
            score += 80
        elif source == 'javascript':
            score += 30
        elif source == 'form_action':
            score += 60
        elif source == 'html_reference':
            score += 20
        elif source == 'text_reference':
            score += 15
        
        # Strong bonus if URL contains '/submit'
        if '/submit' in url_lower:
            score += 80
        
        # Bonus for submission-related keywords in URL
        for keyword in submission_keywords:
            if keyword in url_lower:
                score += 15
        
        # Prefer URLs from same domain
        try:
            page_domain = urlparse(page_url).netloc
            url_domain = urlparse(url).netloc
            if page_domain == url_domain:
                score += 20
        except:
            pass
        
        # Contextual score boost if the surrounding text includes "Submission"
        try:
            url_index = page_text.lower().find(url_lower)
            if url_index > -1:
                context_start = max(0, url_index - 200)
                context_end = min(len(page_text), url_index + len(url) + 200)
                context = page_text[context_start:context_end].lower()
                if 'submission' in context:
                    score += 40
                for keyword in submission_keywords:
                    if keyword in context:
                        score += 10
            # Boost code/pre blocks generally
            if source in ('code_tag',):
                score += 10
        except Exception:
            pass
        
        if score > 0:
            scored_urls.append((score, url, source))
    
    if scored_urls:
        scored_urls.sort(reverse=True, key=lambda x: x[0])
        best_url = scored_urls[0][1]
        best_source = scored_urls[0][2]
        logger.info(f"Submit URL found (score: {scored_urls[0][0]}, source: {best_source}): {best_url}")
        if len(scored_urls) > 1:
            logger.info(f"Alternative URLs found:")
            for score, url, source in scored_urls[1:4]:
                logger.info(f"  - (score: {score}, source: {source}): {url}")
        return best_url
    
    logger.warning("No valid submit URL found")
    return None


async def download_media_file(url: str, media_type: str, index: int) -> Optional[str]:
    """Download media file (image/audio) and return local path"""
    try:
        import aiohttp
        import aiofiles
        
        # Create media directory
        media_dir = Path('data/raw/media')
        media_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        ext = Path(url).suffix or ('.jpg' if media_type == 'image' else '.mp3')
        file_path = media_dir / f"{media_type}_{index}{ext}"
        
        # Handle relative URLs - this is an issue
        if not url.startswith('http'):
            logger.error(f"Invalid URL (relative): {url}")
            return None
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(await response.read())
                    
                    logger.info(f"Downloaded media: {url}")
                    return str(file_path)
                else:
                    logger.warning(f"Media download failed with status {response.status}: {url}")
                    return None
        
    except Exception as e:
        logger.error(f"Failed to download media from {url}: {e}")
        return None


async def fetch_and_render_quiz_page(url: str, max_retries: int = 3) -> dict:
    """
    Fetch and render quiz page with enhanced link detection
    
    Args:
        url: Quiz page URL
        
    Returns:
        Dict containing success status, question, download links, media files, and submit URL
    """
    page = None
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            # Get browser context
            context = await get_browser_context()
            page = await context.new_page()
            
            # Set default timeout
            page.set_default_timeout(PAGE_TIMEOUT)
            
            logger.info(f"Navigating to {url} (attempt {retries + 1}/{max_retries})")
            
            # Navigate to URL
            response = await page.goto(
                url,
                wait_until='domcontentloaded',
                timeout=NAVIGATION_TIMEOUT
            )
            
            # Check response status
            if response and response.status >= 400:
                raise Exception(f"HTTP {response.status} error")
            
            # Wait for content to be ready
            content_ready = await wait_for_content_ready(page)
            
            # Even if content check failed, try to extract what we can
            if not content_ready:
                logger.warning(f"Content readiness check failed on attempt {retries + 1}, but will try to extract data anyway")
            
            # Extract data
            html = await page.content()
            
            # Get page text - handle case where body might be empty
            try:
                page_text = await page.evaluate('() => document.body ? document.body.innerText : ""')
            except Exception:
                page_text = ""
            
            # If we have no text but have HTML, try to extract text from HTML
            if not page_text and html:
                # Strip HTML tags to get text
                import re
                page_text = re.sub(r'<[^>]+>', ' ', html)
                page_text = re.sub(r'\s+', ' ', page_text).strip()
            
            # If still no content, this is a real failure
            if not html or (len(html) < 100 and not page_text):
                raise Exception("Page returned no content")
            
            logger.info(f"Page loaded: HTML size={len(html)}, Text size={len(page_text)}")
            
            # Extract question
            question = extract_question_text(page_text, html) if page_text else None
            
            # Extract submit URL
            submit_url = extract_submit_url(url, html, page_text)
            
            # Extract download links
            download_links = await _extract_download_links(page, url)
            logger.info(f"Found {len(download_links)} download links")
            
            # Extract media files
            media_urls = await _extract_media_files(page, url, html)
            media_files = [{'url': media_url, 'type': 'image' if any(media_url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']) else 'audio'} 
                          for media_url in media_urls]
            # Hint for downstream OCR enforcement
            if any(m['type'] == 'image' for m in media_files):
                logger.info("Image files detected; OCR will be required before answering.")
            
            # Extract structured data from <pre> tags
            structured_data = []
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                for pre_tag in soup.find_all('pre'):
                    pre_id = pre_tag.get('id', '')
                    pre_text = pre_tag.get_text().strip()
                    
                    # Skip submission format examples
                    if 'YOUR_EMAIL' in pre_text or 'YOUR_SECRET' in pre_text:
                        continue
                    
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(pre_text)
                        structured_data.append({
                            'type': 'json',
                            'data': json_data,
                            'id': pre_id,
                            'raw_text': pre_text
                        })
                        logger.info(f"Extracted JSON data from <pre id='{pre_id}'>")
                    except json.JSONDecodeError:
                        # Not valid JSON, keep as formatted text if substantial
                        if len(pre_text) > 20:
                            structured_data.append({
                                'type': 'text',
                                'data': pre_text,
                                'id': pre_id
                            })
                            logger.info(f"Extracted text data from <pre id='{pre_id}'>")
            except Exception as e:
                logger.warning(f"Error extracting structured data: {e}")
            
            logger.info(f"Successfully fetched quiz page")
            
            return {
                'success': True,
                'html': html,
                'text': page_text.strip(),
                'question': question,
                'download_links': download_links,
                'submit_url': submit_url,
                'media_files': media_files,
                'structured_data': structured_data
            }
            
        except PlaywrightTimeout as e:
            last_error = f"Timeout error: {str(e)}"
            logger.error(f"Timeout on attempt {retries + 1}: {e}")
            
        except Exception as e:
            last_error = f"Error: {str(e)}"
            logger.error(f"Error on attempt {retries + 1}: {e}")
        
        finally:
            if page and not page.is_closed():
                try:
                    await page.close()
                except Exception:
                    pass
        
        retries += 1
        if retries < max_retries:
            await asyncio.sleep(RETRY_DELAY)
    
    # All retries exhausted
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return {
        'success': False,
        'html': None,
        'text': None,
        'question': None,
        'download_links': [],
        'submit_url': None,
        'media_files': [],
        'structured_data': [],
        'error': last_error or "Failed after multiple retries"
    }


async def _extract_media_files(page, base_url: str, html_content: str) -> List[str]:
    """
    Extract media file URLs (images and audio) from the page
    Handles both explicit media tags and inline references
    
    Args:
        page: Playwright page object
        base_url: Base URL for resolving relative paths
        html_content: HTML content for parsing
        
    Returns:
        List of absolute media file URLs
    """
    from urllib.parse import urljoin, urlparse
    import re
    
    media_urls = set()
    
    # Supported media extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}
    all_media_extensions = image_extensions | audio_extensions
    
    try:
        # Method 1: Extract from <img> tags
        img_elements = await page.query_selector_all('img')
        for img in img_elements:
            src = await img.get_attribute('src')
            if src:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, src)
                
                # Check if it's a valid media file
                parsed = urlparse(absolute_url)
                path_lower = parsed.path.lower()
                
                if any(path_lower.endswith(ext) for ext in image_extensions):
                    media_urls.add(absolute_url)
                    logger.info(f"Found image in <img> tag: {absolute_url}")
        
        # Method 2: Extract from <audio> and <source> tags
        audio_elements = await page.query_selector_all('audio, source')
        for audio in audio_elements:
            src = await audio.get_attribute('src')
            if src:
                absolute_url = urljoin(base_url, src)
                parsed = urlparse(absolute_url)
                path_lower = parsed.path.lower()
                
                if any(path_lower.endswith(ext) for ext in audio_extensions):
                    media_urls.add(absolute_url)
                    logger.info(f"Found audio file: {absolute_url}")
        
        # Method 3: Extract from HTML content using regex (backup)
        # Find image/audio references in HTML
        media_pattern = r'(?:src|href)=["\']([^"\']+\.(?:' + '|'.join(
            ext.strip('.') for ext in all_media_extensions
        ) + r'))["\']'
        
        matches = re.findall(media_pattern, html_content, re.IGNORECASE)
        for match in matches:
            absolute_url = urljoin(base_url, match)
            media_urls.add(absolute_url)
            logger.info(f"Found media via regex: {absolute_url}")
        
        # Method 4: Look for standalone media file references in text
        # Pattern: filename.ext (without quotes or special chars before/after)
        standalone_pattern = r'\b([\w\-]+\.(?:' + '|'.join(
            ext.strip('.') for ext in all_media_extensions
        ) + r'))\b'
        
        text_matches = re.findall(standalone_pattern, html_content, re.IGNORECASE)
        for filename in text_matches:
            # Try to construct URL relative to current page
            absolute_url = urljoin(base_url, filename)
            media_urls.add(absolute_url)
            logger.info(f"Found media filename: {filename} -> {absolute_url}")
        
    except Exception as e:
        logger.error(f"Error extracting media files: {e}")
    
    media_list = sorted(list(media_urls))
    logger.info(f"Total media files found: {len(media_list)}")
    
    return media_list


async def _extract_download_links(page, base_url: str) -> List[Dict[str, str]]:
    """
    Extract download links for data files (CSV, JSON, etc.) - excluding media files
    
    Args:
        page: Playwright page object
        base_url: Base URL for resolving relative paths
        
    Returns:
        List of dicts with 'url' and 'text' keys
    """
    from urllib.parse import urljoin, urlparse
    import re

    download_links = []

    # Data file extensions (not media)
    data_extensions = {'.csv', '.json', '.xlsx', '.xls', '.txt', '.xml', '.parquet', '.pdf'}
    # Add compressed/archive extensions for matryoshka-type puzzles
    archive_extensions = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tgz'}
    # Media extensions to exclude
    media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg',
                       '.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}

    all_download_exts = data_extensions | archive_extensions

    try:
        # 1) Anchor tag scan (existing behavior)
        links = await page.query_selector_all('a')
        for link in links:
            href = await link.get_attribute('href')
            text = await link.inner_text()
            if href:
                absolute_url = urljoin(base_url, href)
                parsed = urlparse(absolute_url)
                path_lower = parsed.path.lower()

                is_data_file = any(path_lower.endswith(ext) for ext in all_download_exts)
                is_media_file = any(path_lower.endswith(ext) for ext in media_extensions)

                logger.info(f"Link found: {absolute_url} (Data/Archive: {is_data_file}, Media file: {is_media_file})")

                if is_data_file and not is_media_file:
                    download_links.append({
                        'url': absolute_url,
                        'text': (text or '').strip() or parsed.path.split('/')[-1]
                    })
                    logger.info(f"Found data/archive file link: {absolute_url}")

        # 2) HTML content scan for standalone filenames (handles non-anchored references)
        try:
            html_content = await page.content()
        except Exception:
            html_content = ""

        if html_content:
            # Match relative or simple filenames with data/archive extensions appearing anywhere in HTML/code text
            # Examples: "matryoshka.zip", "/assets/data/level14.zip", "files/dataset.csv"
            exts_pattern = '|'.join(re.escape(ext.lstrip('.')) for ext in sorted(all_download_exts, key=len, reverse=True))
            filename_pattern = rf'(?<![A-Za-z0-9_\-/\.])([A-Za-z0-9_\-\/\.]+\.({exts_pattern}))(?![A-Za-z0-9_\-/\.])'

            matches = re.findall(filename_pattern, html_content, flags=re.IGNORECASE)
            # matches returns tuples (full, ext); take the first element as path
            standalone_paths = [m[0] for m in matches]

            # Also scan src/href attributes in raw HTML as a backup
            attr_pattern = rf'(?:href|src)=["\']([^"\']+\.({exts_pattern}))["\']'
            attr_matches = re.findall(attr_pattern, html_content, flags=re.IGNORECASE)
            standalone_paths.extend([m[0] for m in attr_matches])

            # Deduplicate while preserving order
            seen_paths = set()
            ordered_paths = []
            for p in standalone_paths:
                p_clean = p.strip()
                if not p_clean or p_clean in seen_paths:
                    continue
                # Ignore template literals/unrendered placeholders
                if '${' in p_clean or '{' in p_clean or '`' in p_clean:
                    continue
                seen_paths.add(p_clean)
                ordered_paths.append(p_clean)

            for rel in ordered_paths:
                absolute_url = urljoin(base_url, rel)
                parsed = urlparse(absolute_url)
                path_lower = parsed.path.lower()
                # Skip media files and only keep data/archive
                if any(path_lower.endswith(ext) for ext in media_extensions):
                    continue
                if any(path_lower.endswith(ext) for ext in all_download_exts):
                    # Avoid duplicates with anchors already collected
                    if not any(d['url'] == absolute_url for d in download_links):
                        download_links.append({
                            'url': absolute_url,
                            'text': parsed.path.split('/')[-1]
                        })
                        logger.info(f"Found data/archive via HTML scan: {absolute_url}")

        # 3) Code/pre blocks heuristic for dynamic snippets (e.g., showing path pieces)
        # If a code/pre shows a full URL or relative path, try to include it.
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content or "", 'html.parser')
            for tag in soup.find_all(['code', 'pre']):
                code_text = (tag.get_text() or '').strip()
                if not code_text:
                    continue
                # Direct http(s) URLs ending with desired extensions
                url_matches = re.findall(r'(https?://[^\s\'"]+)', code_text, re.IGNORECASE)
                for u in url_matches:
                    parsed = urlparse(u)
                    path_lower = parsed.path.lower()
                    if any(path_lower.endswith(ext) for ext in all_download_exts) and not any(path_lower.endswith(ext) for ext in media_extensions):
                        if not any(d['url'] == u for d in download_links):
                            download_links.append({'url': u, 'text': parsed.path.split('/')[-1]})
                            logger.info(f"Found data/archive in code block: {u}")
                # Relative file references inside code/pre
                rel_matches = re.findall(r'(?<![A-Za-z0-9_\-/\.])([A-Za-z0-9_\-\/\.]+\.(?:' + exts_pattern + r'))(?![A-Za-z0-9_\-/\.])', code_text, re.IGNORECASE)
                for rel in rel_matches:
                    # When using groups, rel can be a tuple; take the first element if so
                    rel_path = rel[0] if isinstance(rel, tuple) else rel
                    absolute_url = urljoin(base_url, rel_path)
                    parsed = urlparse(absolute_url)
                    path_lower = parsed.path.lower()
                    if any(path_lower.endswith(ext) for ext in all_download_exts) and not any(path_lower.endswith(ext) for ext in media_extensions):
                        if not any(d['url'] == absolute_url for d in download_links):
                            download_links.append({'url': absolute_url, 'text': parsed.path.split('/')[-1]})
                            logger.info(f"Found data/archive in code/pre: {absolute_url}")
        except Exception as e:
            logger.debug(f"Code/pre scan skipped: {e}")

    except Exception as e:
        logger.error(f"Error extracting download links: {e}")

    return download_links


# Cleanup function for graceful shutdown
async def cleanup():
    """Cleanup resources on shutdown"""
    await close_browser()

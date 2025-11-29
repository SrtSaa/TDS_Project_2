"""
Image Processing Module - Extracts text from images using OCR
"""
import logging
import httpx
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import base64
import re
import os
import time

logger = logging.getLogger(__name__)


async def extract_text_from_images(
    media_files: List[Dict[str, str]],
    processed_files: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract text from images using OCR
    
    Args:
        media_files: List of media file dictionaries with 'url' and 'type'
        processed_files: List of already processed/downloaded files
        
    Returns:
        List of dictionaries containing extracted text and metadata
    """
    extractions = []
    
    # Find image files
    image_files = []
    
    # Check media files
    for media in media_files:
        if media.get('type', '').startswith('image/'):
            image_files.append({
                'url': media['url'],
                'source': 'media'
            })
    
    # Build set of candidate image paths from processed_files
    for pfile in processed_files:
        # Generic keys used by data_processor
        for key in ['processed_file', 'original_file']:
            fp = pfile.get(key)
            if fp and any(fp.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
                image_files.append({
                    'path': fp,
                    'url': pfile.get('original_url', ''),
                    'source': 'processed'
                })
                break

    # Expand media_files detection (handle type='image', 'img', or missing but URL extension matches)
    for media in (media_files or []):
        url = media.get('url', '')
        media_type = media.get('type', '').lower()
        if (
            media_type.startswith('image') or
            media_type in ('img', 'image') or
            any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'])
        ):
            # Attempt to map to processed file by basename
            basename = url.split('?')[0].rsplit('/', 1)[-1]
            local_match = next(
                (pf.get('processed_file') for pf in processed_files
                 if pf.get('processed_file') and os.path.basename(pf['processed_file']) == f"{Path(basename).stem}_processed{Path(basename).suffix}"),
                None
            )
            image_files.append({
                'url': url,
                'path': local_match,
                'source': 'media'
            })
    
    if not image_files:
        logger.info("No image files found to extract text from")
        return []
    
    # Deduplicate (prefer entries with local path) BEFORE logging count
    dedup = {}
    for entry in image_files:
        key = (os.path.basename(entry.get('path') or '') or os.path.basename(entry.get('url','')) or entry.get('url','')).lower()
        existing = dedup.get(key)
        if existing:
            # Prefer one with local path
            if not existing.get('path') and entry.get('path'):
                dedup[key] = entry
        else:
            dedup[key] = entry
    image_files = list(dedup.values())
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, img_info in enumerate(image_files):
        try:
            # Get image data
            image_data = None
            image_path = img_info.get('path')
            image_url = img_info.get('url', '')
            
            if image_path and Path(image_path).exists():
                # Read from local file
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                logger.info(f"Read image from local file: {image_path}")
            elif image_url:
                # Download image
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    image_data = response.content
                logger.info(f"Downloaded image from URL: {image_url}")
            
            if not image_data:
                logger.warning(f"Could not get image data for image {idx}")
                continue
            
            # Extract text using OCR via LLM vision
            extracted_text = await _extract_text_with_vision(image_data, image_url)
            
            if extracted_text:
                extractions.append({
                    'url': image_url,
                    'path': image_path,
                    'text': extracted_text,
                    'success': True
                })
                logger.info(f"Successfully extracted text from image {idx+1}: {len(extracted_text)} characters")
            else:
                extractions.append({
                    'url': image_url,
                    'path': image_path,
                    'text': '',
                    'success': False,
                    'error': 'No text extracted'
                })
                logger.warning(f"No text extracted from image {idx+1}")
        
        except Exception as e:
            logger.error(f"Error processing image {idx+1}: {e}")
            extractions.append({
                'url': img_info.get('url', ''),
                'path': img_info.get('path', ''),
                'text': '',
                'success': False,
                'error': str(e)
            })
    
    return extractions


def _is_meaningful_text(text: str) -> bool:
    """Generic validation for extracted OCR text."""
    if not text:
        return False
    # Require at least 15 non-whitespace chars and at least 2 distinct words
    cleaned = re.sub(r'\s+', ' ', text.strip())
    if len(cleaned) < 15:
        return False
    words = set(w.lower() for w in re.findall(r'[A-Za-z0-9]+', cleaned))
    return len(words) >= 2


async def _extract_text_with_vision(image_data: bytes, image_url: str) -> str:
    """
    Extract text from image using LLM vision capabilities (two-pass with validation).
    """
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_type = "image/png"
    if image_data[:2] == b'\xff\xd8':
        image_type = "image/jpeg"
    elif image_data[:4] == b'GIF8':
        image_type = "image/gif"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("AIPIPE_TOKEN")
    if not api_key:
        logger.error("OPENROUTER_API_KEY or AIPIPE_TOKEN not found in environment")
        return ""

    async def _vision_call(prompt: str) -> str:
        max_attempts = 4
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        "https://aipipe.org/openrouter/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "google/gemini-2.0-flash-exp:free",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_base64}"}}
                                    ]
                                }
                            ],
                            "temperature": 0.0
                        }
                    )
                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        wait = min(6.0, 2 ** attempt * 0.5)
                        logger.warning(f"Vision API status {resp.status_code}; retrying in {wait:.1f}s (attempt {attempt}/{max_attempts})")
                        await asyncio.sleep(wait)
                        last_error = f"status {resp.status_code}"
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429 or 500 <= code < 600:
                    wait = min(6.0, 2 ** attempt * 0.5)
                    logger.warning(f"OCR retry after {code}; sleeping {wait:.1f}s (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(wait)
                    last_error = f"status {code}"
                    continue
                logger.error(f"Vision API HTTP error: {e}")
                return ""
            except Exception as e:
                logger.error(f"Vision API error: {e}")
                last_error = str(e)
                break
        if last_error:
            logger.error(f"Vision API exhausted retries ({last_error})")
        return ""

    # Pass 1: exhaustive extraction
    pass1_prompt = (
        "Extract EVERY piece of text in this image exactly as it appears. "
        "Include numbers, punctuation, code, labels, headings, footers. "
        "Preserve line breaks. Return ONLY raw text."
    )
    text1 = await _vision_call(pass1_prompt)

    if _is_meaningful_text(text1):
        logger.info(f"OCR pass 1 succeeded (len={len(text1)})")
        return text1

    # Pass 2: fallback / normalization
    pass2_prompt = (
        "Retry OCR. Return all readable text in plain lines. Do not summarize or omit any element. "
        "If something looks faint include it. ONLY output text."
    )
    text2 = await _vision_call(pass2_prompt)

    if _is_meaningful_text(text2):
        logger.info(f"OCR pass 2 succeeded (len={len(text2)})")
        return text2

    logger.warning("OCR failed to produce meaningful text after two passes")
    return text2 or text1 or ""

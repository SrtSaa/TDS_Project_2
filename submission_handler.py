"""
Submission Handler Module
Handles quiz answer submission and response processing
"""

import httpx
import logging
from typing import Dict, Any, Optional
import time
import asyncio

logger = logging.getLogger(__name__)

# Constants
SUBMISSION_TIMEOUT = 30.0  # 30 seconds timeout for submission
MAX_SUBMISSION_RETRIES = 2  # Retry submission on network errors


async def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    current_url: str,
    answer: Any,
    timeout: float = SUBMISSION_TIMEOUT,
    require_ocr: bool = False,
    ocr_verified: bool = True
) -> Dict[str, Any]:
    """
    Submit quiz answer to the server
    
    Args:
        submit_url: URL to submit the answer
        email: User email
        secret: Secret key
        current_url: Current quiz URL being solved
        answer: The answer to submit
        timeout: Request timeout in seconds
        require_ocr: Whether OCR verification is required
        ocr_verified: Whether OCR has been verified
        
    Returns:
        Dict with submission result:
        - success: bool
        - correct: bool (if successful)
        - next_url: str (if provided)
        - reason: str (if incorrect)
        - response: dict (full response)
        - error: str (if failed)
    """
    # OCR enforcement
    if require_ocr and not ocr_verified:
        return {
            'success': False,
            'error': 'OCR verification failed or missing before submission.'
        }
    
    payload = {
        "email": email,
        "secret": secret,
        "url": current_url,
        "answer": answer
    }
    
    logger.info(f"Submitting answer to {submit_url}")
    
    for attempt in range(MAX_SUBMISSION_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                print(submit_url)
                for item in payload:
                    print(f"{item}: {payload[item]}")
                response = await client.post(submit_url, json=payload)
                response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    result = response.json()
                except Exception as json_error:
                    logger.error(f"Failed to parse JSON response: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    return {
                        'success': False,
                        'error': f'Invalid JSON response from server: {response.text[:200]}'
                    }
                
                is_correct = result.get('correct', False)
                next_url = result.get('url')
                reason = result.get('reason')
                
                logger.info(f"Submission result: correct={is_correct}")
                if next_url:
                    logger.info(f"Next URL: {next_url}")
                if reason:
                    logger.warning(f"Failure reason: {reason}")
                
                return {
                    'success': True,
                    'correct': is_correct,
                    'next_url': next_url,
                    'reason': reason,
                    'response': result
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error on submission attempt {attempt + 1}: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text[:500]}")
            if attempt == MAX_SUBMISSION_RETRIES - 1:
                return {
                    'success': False,
                    'error': f'HTTP {e.response.status_code}: {e.response.text[:200]}'
                }
            await asyncio.sleep(1)
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout on submission attempt {attempt + 1}")
            if attempt == MAX_SUBMISSION_RETRIES - 1:
                return {
                    'success': False,
                    'error': 'Submission timeout'
                }
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            if attempt == MAX_SUBMISSION_RETRIES - 1:
                return {
                    'success': False,
                    'error': str(e)
                }
            await asyncio.sleep(1)
    
    return {
        'success': False,
        'error': 'Failed after multiple retries'
    }


async def submit_dummy_answer(
    submit_url: str,
    email: str,
    secret: str,
    current_url: str,
    expected_type: str = "string"
) -> Dict[str, Any]:
    """
    Submit a dummy fallback answer when exceptions occur
    
    Args:
        submit_url: URL to submit the answer
        email: User email
        secret: Secret key
        current_url: Current quiz URL
        expected_type: Expected answer type (string, number, boolean, dict)
        
    Returns:
        Dict with submission result
    """
    logger.warning(f"Submitting dummy answer for {current_url}")
    
    # Generate type-appropriate dummy answer
    if expected_type == "number":
        dummy_value = 0
    elif expected_type == "boolean":
        dummy_value = False
    elif expected_type == "dict":
        dummy_value = {"answer": "dummy"}
    else:
        dummy_value = "DUMMY_ANSWER"
    
    return await submit_answer(
        submit_url=submit_url,
        email=email,
        secret=secret,
        current_url=current_url,
        answer=dummy_value,
        timeout=10.0  # Shorter timeout for dummy submission
    )


def should_retry(
    submission_result: Dict[str, Any],
    elapsed_time: float,
    total_time_limit: float
) -> bool:
    """
    Determine if we should retry the current quiz
    
    Args:
        submission_result: Result from submit_answer
        elapsed_time: Time elapsed since start
        total_time_limit: Total time limit for entire process
        
    Returns:
        True if should retry
    """
    if not submission_result.get('success'):
        return False
    
    if submission_result.get('correct'):
        return False
    
    # Check if at least 50% time remains
    time_remaining = total_time_limit - elapsed_time
    retry_threshold = total_time_limit * 0.5
    
    should_retry_flag = time_remaining >= retry_threshold
    
    if should_retry_flag:
        logger.info(f"Will retry: {time_remaining:.1f}s remaining")
    else:
        logger.info(f"Skipping retry: {time_remaining:.1f}s remaining")
    
    return should_retry_flag

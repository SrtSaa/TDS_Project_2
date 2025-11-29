"""
Answer Generator Module
Fully LLM-driven answer generation using code execution capabilities.
Supports text, image, and audio inputs.
"""

import json
import logging
import base64
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from io import StringIO
import httpx
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import tempfile
import sys
from contextlib import redirect_stdout, redirect_stderr
import re
import ast

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
LLM_AVAILABLE = bool(AIPIPE_TOKEN)

if not LLM_AVAILABLE:
    logger.warning("AIPIPE_TOKEN not set. LLM functionality will be unavailable.")


def extract_javascript_from_html(html: str) -> Optional[str]:
    """
    Extract JavaScript code from HTML script tags.
    
    Args:
        html: HTML content containing script tags
        
    Returns:
        Extracted JavaScript code or None
    """
    if not html:
        return None
    
    # Find all script tags
    script_pattern = r'<script[^>]*>(.*?)</script>'
    scripts = re.findall(script_pattern, html, re.DOTALL | re.IGNORECASE)
    
    # Filter out scripts that are just URL replacements or utilities
    relevant_scripts = []
    for script in scripts:
        script = script.strip()
        # Skip empty scripts and simple DOM manipulation
        if not script or 'querySelectorAll' in script or 'Replace URL' in script:
            continue
        relevant_scripts.append(script)
    
    if not relevant_scripts:
        return None
    
    # Combine all relevant scripts
    return '\n\n'.join(relevant_scripts)


def convert_javascript_to_python(js_code: str) -> str:
    """
    Convert JavaScript code to executable Python code.
    Handles common patterns like function declarations, let/const, etc.
    
    Args:
        js_code: JavaScript code to convert
        
    Returns:
        Python-equivalent code
    """
    python_code = js_code
    
    # Convert function declarations: function name(...) { ... }
    python_code = re.sub(
        r'function\s+(\w+)\s*\((.*?)\)\s*{',
        r'def \1(\2):',
        python_code
    )
    
    # Convert arrow functions: const name = (...) => { ... }
    python_code = re.sub(
        r'const\s+(\w+)\s*=\s*\((.*?)\)\s*=>\s*{',
        r'def \1(\2):',
        python_code
    )
    
    # Convert let/const/var declarations to simple assignments
    python_code = re.sub(r'\b(let|const|var)\s+', '', python_code)
    
    # Remove semicolons
    python_code = python_code.replace(';', '')
    
    # Fix indentation
    lines = python_code.split('\n')
    processed_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
            
        # Decrease indent for closing braces
        if '}' in stripped:
            indent_level = max(0, indent_level - 1)
            stripped = stripped.replace('}', '').strip()
            if not stripped:
                continue
        
        # Add proper indentation
        if stripped:
            processed_lines.append('    ' * indent_level + stripped)
        
        # Increase indent after function/if/for/while
        if stripped.endswith(':') or 'def ' in stripped:
            indent_level += 1
    
    return '\n'.join(processed_lines)


def execute_javascript_as_python(js_code: str) -> Optional[Any]:
    """
    Execute JavaScript code by converting it to Python.
    
    Args:
        js_code: JavaScript code to execute
        
    Returns:
        Result of the execution or None if failed
    """
    try:
        # Convert JS to Python
        python_code = convert_javascript_to_python(js_code)
        logger.info(f"Converted JavaScript to Python:\n{python_code}")
        
        # Create execution environment
        exec_globals = {
            'Math': type('Math', (), {
                'floor': lambda x: int(x),
                'ceil': lambda x: int(x) + (1 if x != int(x) else 0),
                'round': round,
                'abs': abs,
                'sqrt': lambda x: x ** 0.5,
                'pow': pow,
            }),
            '__builtins__': __builtins__,
        }
        
        # Execute the code
        exec(python_code, exec_globals)
        
        # Find all function definitions
        functions = [name for name in exec_globals if callable(exec_globals[name]) and not name.startswith('_')]
        
        if not functions:
            logger.warning("No functions found after execution")
            return None
        
        # Execute each function that takes no arguments and get results
        results = []
        for func_name in functions:
            func = exec_globals[func_name]
            try:
                # Check if function takes no arguments
                import inspect
                sig = inspect.signature(func)
                if len(sig.parameters) == 0:
                    result = func()
                    results.append(result)
                    logger.info(f"Function {func_name}() returned: {result}")
            except Exception as e:
                logger.warning(f"Could not execute {func_name}: {e}")
        
        # Return the last/most relevant result
        return results[-1] if results else None
        
    except Exception as e:
        logger.error(f"JavaScript execution error: {e}")
        return None


class AnswerGenerator:
    """Generates quiz answers using fully LLM-driven approach"""
    
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        """
        Initialize answer generator
        
        Args:
            model: LLM model to use (default: openai/gpt-4o-mini for multimodal)
        """
        self.model = model
        self.max_tokens = 8000
        self.max_json_size = 1_000_000  # 1MB limit
        self.api_url = "https://aipipe.org/openrouter/v1/chat/completions"
        self.timeout = 180.0  # 3 minutes timeout for LLM calls
        
        # Supported media types
        self.supported_image_types = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        self.supported_audio_types = {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}
    
    async def generate_answer(
        self,
        instructions: str,
        processed_data: List[Dict[str, Any]],
        page_html: Optional[str] = None,
        media_files: Optional[List[Dict[str, Any]]] = None,
        image_text_extractions: Optional[List[Dict[str, Any]]] = None,
        failure_reason: Optional[str] = None,
        structured_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer based on instructions and processed data
        
        Args:
            instructions: Quiz instructions/question text
            processed_data: List of processed data files with metadata
            page_html: Optional HTML content for additional context
            media_files: Optional list of image/audio file paths
            image_text_extractions: List of text extracted from images via OCR
            failure_reason: Previous failure reason for retry attempts
            
        Returns:
            Dict with 'success', 'answer', 'answer_type', 'reasoning', 'error'
        """
        try:
            if not LLM_AVAILABLE:
                return {
                    'success': False,
                    'error': 'LLM not configured. Set AIPIPE_TOKEN environment variable.'
                }
            
            if failure_reason:
                logger.warning(f"Retry mode - Previous failure: {failure_reason}")
            
            # Process media files first with local fallback
            processed_media = []
            if media_files:
                try:
                    processed_media = await self._process_media_files(media_files, processed_data)
                except Exception as e:
                    logger.warning(f"Local media pre-processing failed: {e}")
            
            # Build user message with data context
            user_parts = []
            
            # Add failure context if retrying
            if failure_reason:
                user_parts.append(f"PREVIOUS ATTEMPT FAILED: {failure_reason}")
                user_parts.append("Please provide a different, more accurate answer.\n")
            
            # Add instructions
            if instructions:
                user_parts.append(f"Instructions:\n{instructions}")
            
            # Build context with image text extractions
            context_parts = []
            
            # Add image text extractions if available
            if image_text_extractions:
                context_parts.append("=== TEXT EXTRACTED FROM IMAGES ===")
                for idx, extraction in enumerate(image_text_extractions):
                    if extraction.get('success') and extraction.get('text'):
                        context_parts.append(f"\nImage {idx+1} ({extraction.get('url', 'unknown')}):")
                        context_parts.append(extraction['text'])
                        context_parts.append("")
                context_parts.append("=== END OF IMAGE TEXT ===\n")
            
            # Add processed data
            if processed_data:
                context_parts.append("\nProcessed Data:")
                
                # Convert to list if it's a dict (for backwards compatibility)
                if isinstance(processed_data, dict):
                    data_items = processed_data.items()
                elif isinstance(processed_data, list):
                    # Create enumerated items from list
                    data_items = [(f"file_{idx}", item) for idx, item in enumerate(processed_data)]
                else:
                    logger.warning(f"Unexpected processed_data type: {type(processed_data)}")
                    data_items = []
                
                for file_key, file_info in data_items:
                    file_type = file_info.get('type', 'unknown')
                    summary = file_info.get('summary', 'No summary')
                    context_parts.append(f"\n{file_key} ({file_type}):\n{summary}")
            
            # Process media files (images, videos)
            if media_files:
                logger.info(f"Processing {len(media_files)} media files")
                for idx, media in enumerate(media_files):
                    media_url = media.get('url', '')
                    media_type = media.get('type', 'unknown')
                    local_path = media.get('local_path')
                    if media_type in ['img', 'image'] or any(str(media_url).lower().endswith(ext) for ext in self.supported_image_types):
                        try:
                            import mimetypes
                            # Prefer local path if available
                            if local_path and os.path.exists(local_path):
                                with open(local_path, 'rb') as f:
                                    image_data = f.read()
                                logger.info(f"Using local image copy: {local_path}")
                            else:
                                import httpx
                                async with httpx.AsyncClient(timeout=30.0) as client:
                                    response = await client.get(media_url)
                                    response.raise_for_status()
                                    image_data = response.content
                                raw_dir = Path("data/raw")
                                raw_dir.mkdir(parents=True, exist_ok=True)
                                from urllib.parse import urlparse
                                parsed = urlparse(media_url)
                                file_ext = Path(parsed.path).suffix or '.png'
                                local_path = raw_dir / f"media_{idx}{file_ext}"
                                with open(local_path, 'wb') as f:
                                    f.write(image_data)
                                logger.info(f"Downloaded image: {local_path} ({len(image_data)} bytes)")
                            mime_type, _ = mimetypes.guess_type(str(local_path))
                            if not mime_type:
                                mime_type = 'image/png'
                            encoded = base64.b64encode(image_data).decode('utf-8')
                            user_parts.append(f"\nImage {idx + 1} (from {media_url or local_path}):")
                            user_parts.append("Analyze this image carefully to answer the question.")
                            if not hasattr(generate_quiz_answer, '_image_parts'):
                                generate_quiz_answer._image_parts = []
                            generate_quiz_answer._image_parts.append({
                                'type': 'image',
                                'mime_type': mime_type,
                                'data': encoded,
                                'url': media_url or str(local_path)
                            })
                        except Exception as e:
                            logger.error(f"Error processing image {media_url or local_path}: {e}")
                            user_parts.append(f"\nImage {idx + 1}: {media_url or local_path} (load failed)")
                    else:
                        user_parts.append(f"\nMedia {idx + 1} ({media_type}): {media_url}")
            
            user_message = "\n\n".join(user_parts)
            
            # Prepare data context for LLM - handle both list and dict
            if isinstance(processed_data, list):
                data_context = self._prepare_data_context(processed_data)
            elif isinstance(processed_data, dict):
                # Convert dict to list format
                data_list = [{'index': k, **v} for k, v in processed_data.items()]
                data_context = self._prepare_data_context(data_list)
            else:
                data_context = {'files': [], 'total_files': 0}
            
            # Collect extracted file contents (general, scalable)
            extracted_texts = []
            try:
                for item in processed_data or []:
                    path = item.get('processed_file') or item.get('original_file')
                    if not path or not os.path.exists(path):
                        continue
                    # Only read small manageable snippets to keep prompt size reasonable
                    if Path(path).suffix.lower() in {'.txt', '.md', '.log', '.csv', '.json'}:
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            try:
                                with open(path, 'r', encoding='latin-1') as f:
                                    content = f.read()
                            except Exception:
                                content = ""
                        if content:
                            # Truncate to avoid oversized prompts
                            snippet = content[:4000]
                            extracted_texts.append({
                                'file_name': os.path.basename(path),
                                'length': len(content),
                                'snippet': snippet
                            })
            except Exception as e:
                logger.warning(f"Could not aggregate extracted file contents: {e}")

            # Enforce OCR availability before answering when images present (consider local media)
            image_present = any(
                (m.get('type','').startswith('image') or m.get('type') in ('img','image'))
                or any(str(m.get('url','')).lower().endswith(ext) for ext in self.supported_image_types)
                for m in (media_files or [])
            ) or any(pm.get('type') == 'image' for pm in processed_media)
            if image_present:
                valid_extractions = [
                    e for e in (image_text_extractions or [])
                    if e.get('success') and e.get('text') and len(e.get('text').strip()) >= 15
                ]
                if not valid_extractions:
                    # Fallback inline OCR attempt (generic)
                    logger.warning("Insufficient OCR text; attempting inline OCR fallback")
                    try:
                        fallback_extractions = await self._inline_ocr_fallback(media_files, processed_media)
                        if fallback_extractions:
                            if image_text_extractions is None:
                                image_text_extractions = []
                            image_text_extractions.extend(fallback_extractions)
                            valid_extractions = [
                                e for e in image_text_extractions
                                if e.get('success') and e.get('text') and len(e.get('text').strip()) >= 15
                            ]
                    except Exception as ocr_err:
                        logger.error(f"Inline OCR fallback failed: {ocr_err}")
                    if not valid_extractions:
                        return {
                            'success': False,
                            'error': 'Missing or insufficient OCR text for images. Extract text before generating answer.'
                        }

            # Promote OCR text blocks into extracted_texts for unified handling
            if image_text_extractions:
                for e in image_text_extractions:
                    if e.get('text'):
                        extracted_texts.append({
                            'file_name': e.get('url','image'),
                            'length': len(e['text']),
                            'snippet': e['text'][:4000]
                        })
            
            # Fast path: handle ML linear regression if structured_data contains X and y (general)
            try:
                if structured_data:
                    for block in structured_data:
                        if block.get('type') == 'json' and isinstance(block.get('data'), dict):
                            data_obj = block['data']
                            if 'X' in data_obj and 'y' in data_obj:
                                import numpy as np
                                X_list = data_obj['X']
                                y_list = data_obj['y']
                                X = np.array([row[0] if isinstance(row, (list, tuple)) else row for row in X_list], dtype=float)
                                y = np.array(y_list, dtype=float)
                                mean_x = X.mean()
                                mean_y = y.mean()
                                cov = np.sum((X - mean_x) * (y - mean_y))
                                var = np.sum((X - mean_x) ** 2)
                                slope = cov / var if var != 0 else 0.0
                                intercept = mean_y - slope * mean_x
                                # Generic: if an explicit X target exists in instructions, try to parse it; else leave to LLM
                                import re as _re
                                m = _re.search(r'\bX\s*=\s*([0-9]+(?:\.[0-9]+)?)', (instructions or ''), flags=_re.IGNORECASE)
                                if m:
                                    x_new = float(m.group(1))
                                    y_pred = slope * x_new + intercept
                                    answer_value = y_pred
                                    # If rounding is mentioned, apply it generally
                                    round_m = _re.search(r'round[^0-9]*([0-9]+)\s*decimal', (instructions or ''), flags=_re.IGNORECASE)
                                    if round_m:
                                        answer_value = round(answer_value, int(round_m.group(1)))
                                    return {
                                        'success': True,
                                        'answer': float(answer_value),
                                        'answer_type': 'number',
                                        'reasoning': 'Computed linear regression from provided training data and predicted at specified X.',
                                        'code_executed': 'direct_regression',
                                        'metadata': {'slope': slope, 'intercept': intercept, 'x_new': x_new}
                                    }
            except Exception as e:
                logger.warning(f"Direct ML fast-path failed, falling back to LLM: {e}")

            # Get LLM to generate Python code to solve the problem, with extracted text context
            code_response = await self._generate_solution_code(
                instructions=instructions,
                data_context=data_context,
                processed_media=processed_media,
                page_html=page_html,
                failure_reason=failure_reason,
                extracted_texts=extracted_texts  # pass collected text contexts
            )
            
            if not code_response['success']:
                return code_response
            
            # Execute the generated code
            execution_result = await self._execute_generated_code(
                code=code_response['code'],
                data_files=processed_data
            )
            
            if not execution_result['success']:
                return execution_result
            
            # Validate and format final answer
            final_answer = self._format_and_validate_answer(
                answer=execution_result['result'],
                answer_type=execution_result.get('answer_type', 'auto')
            )
            
            logger.info(f"Answer generated: type={final_answer['type']}")
            
            return {
                'success': True,
                'answer': final_answer['value'],
                'answer_type': final_answer['type'],
                'reasoning': code_response.get('reasoning', ''),
                'code_executed': code_response['code'],
                'metadata': {
                    'json_size': len(json.dumps(final_answer['value'])),
                    'is_base64': final_answer['type'] == 'base64',
                    'media_processed': len(processed_media)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_media_files(self, media_files: List[str], processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process image and audio files for multimodal analysis with local fallback."""
        processed = []

        # Build lookup of downloaded/processed files by basename
        file_lookup = {}
        try:
            for item in (processed_data or []):
                for key in ['processed_file', 'original_file']:
                    path = item.get(key)
                    if path and os.path.exists(path):
                        file_lookup[os.path.basename(path)] = path
        except Exception:
            pass

        for media_item in (media_files or []):
            if isinstance(media_item, str):
                media_dict = {'url': media_item}
            elif isinstance(media_item, dict):
                media_dict = media_item
            else:
                logger.warning(f"Unexpected media item type: {type(media_item)}")
                continue

            # Derive candidate path
            file_path = media_dict.get('local_path') or media_dict.get('path')
            if not file_path and media_dict.get('url'):
                # Try basename matching with downloaded files
                basename = os.path.basename(media_dict['url'].split('?')[0])
                if basename in file_lookup:
                    file_path = file_lookup[basename]

            # If still missing, ignore for local processing (download happens elsewhere)
            if not file_path or not os.path.exists(file_path):
                logger.debug(f"Media local path unresolved for {media_dict.get('url')}")
                continue

            file_ext = Path(file_path).suffix.lower()
            try:
                if file_ext in self.supported_image_types:
                    media_data = await self._process_image(file_path)
                    media_data['source'] = 'local'
                    media_data['original_url'] = media_dict.get('url', '')
                    processed.append(media_data)
                elif file_ext in self.supported_audio_types:
                    media_data = await self._process_audio(file_path)
                    media_data['source'] = 'local'
                    media_data['original_url'] = media_dict.get('url', '')
                    processed.append(media_data)
            except Exception as e:
                logger.error(f"Error processing local media file {file_path}: {e}")

        return processed
    
    async def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image file and convert to base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'image/jpeg'
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            return {
                'type': 'image',
                'mime_type': mime_type,
                'data': base64_image,
                'file_name': os.path.basename(image_path)
            }
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'type': 'image',
                'error': str(e),
                'file_name': os.path.basename(image_path)
            }
    
    async def _process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file - transcribe using Whisper API"""
        try:
            transcription = await self._transcribe_audio(audio_path)
            
            return {
                'type': 'audio',
                'transcription': transcription,
                'file_name': os.path.basename(audio_path)
            }
        
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return {
                'type': 'audio',
                'error': str(e),
                'file_name': os.path.basename(audio_path)
            }
    
    async def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper API via AIpipe"""
        try:
            headers = {
                "authorization": f"Bearer {AIPIPE_TOKEN}",
            }
            
            with open(audio_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(audio_path), f, 'audio/mpeg')
                }
                data = {
                    'model': 'openai/whisper-1'
                }
                
                audio_url = "https://aipipe.org/openrouter/v1/audio/transcriptions"
                
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(
                        audio_url,
                        headers=headers,
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get('text', '')
                    else:
                        return f"[Audio transcription unavailable: {response.status_code}]"
        
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return f"[Audio transcription error: {str(e)}]"
    
    def _prepare_data_context(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data context with file paths and metadata"""
        context = {
            'files': [],
            'total_files': len(processed_data)
        }
        
        for idx, file_info in enumerate(processed_data):
            file_path = file_info.get('processed_file', f'file_{idx}')
            
            file_context = {
                'index': idx,
                'path': file_path,
                'original_url': file_info.get('original_url', ''),
                'file_name': os.path.basename(file_path),
                'steps_performed': file_info.get('steps_performed', [])
            }
            
            # Add basic file info without loading entire dataset
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=5)
                    file_context['format'] = 'csv'
                    file_context['columns'] = list(df.columns)
                    file_context['sample_rows'] = len(df)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path, lines=True, nrows=5)
                    file_context['format'] = 'json'
                    file_context['columns'] = list(df.columns)
                    file_context['sample_rows'] = len(df)
            except Exception:
                pass
            
            context['files'].append(file_context)
        
        return context
    
    async def _generate_solution_code(
        self,
        instructions: str,
        data_context: Dict[str, Any],
        processed_media: List[Dict[str, Any]],
        page_html: Optional[str],
        failure_reason: Optional[str] = None,
        extracted_texts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Ask LLM to generate Python code to solve the problem"""
        
        # Check if this is a JavaScript extraction task (not just any script tag)
        # Only trigger JS extraction if there's actual computation code
        is_js_task = False
        if page_html:
            js_code = extract_javascript_from_html(page_html)
            # Check if extracted JS contains actual functions/computation
            if js_code and ('function' in js_code or 'def ' in js_code or 'return' in js_code):
                # Exclude simple DOM manipulation
                if 'querySelectorAll' not in js_code and 'Replace URL' not in js_code:
                    is_js_task = True
        
        if is_js_task:
            logger.info("Detected JavaScript extraction task - attempting direct execution")
            
            result = execute_javascript_as_python(js_code)
            
            if result is not None:
                logger.info(f"JavaScript execution result: {result}")
                return {
                    'success': True,
                    'code': f'# JavaScript execution result\nanswer = {result}',
                    'reasoning': 'Extracted and executed JavaScript from HTML',
                    'answer_type': 'number' if isinstance(result, (int, float)) else 'string'
                }
            else:
                logger.warning("JavaScript execution returned None, falling back to LLM")
        
        # Build media context
        media_text = []
        image_contents = []
        
        for media in processed_media:
            if media['type'] == 'image' and 'data' in media:
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media['mime_type']};base64,{media['data']}"
                    }
                })
                media_text.append(f"[Image: {media['file_name']}]")
            elif media['type'] == 'audio' and 'transcription' in media:
                media_text.append(f"[Audio from {media['file_name']}]: {media['transcription']}")
        
        media_context = "\n".join(media_text) if media_text else "No media files."
        
        failure_context = ""
        if failure_reason:
            failure_context = f"""

IMPORTANT - PREVIOUS ATTEMPT FAILED:
Reason: {failure_reason}

Please analyze what went wrong and correct your approach.
"""

        # NEW: Hard constraint to use ONLY extracted text when present
        strict_text_constraint = """
STRICT CONSTRAINTS:
- You MUST derive the answer exclusively from the provided extracted file contents/snippets.
- Do NOT use any external knowledge or assumptions beyond the provided text.
- If the answer cannot be derived from the text, return a clear error in 'answer' explaining insufficient data.
"""

        # Build extracted text context section (general)
        extracted_section = ""
        if extracted_texts:
            try:
                entries = []
                for entry in extracted_texts[:10]:
                    entries.append({
                        'file': entry['file_name'],
                        'length': entry['length'],
                        'snippet': entry['snippet']
                    })
                extracted_section = "\n\nEXTRACTED FILE CONTENTS (truncated):\n" + json.dumps(entries, indent=2)
            except Exception:
                extracted_section = "\n\nEXTRACTED FILE CONTENTS: [unavailable]"

        prompt = f"""You are an expert data analyst and Python programmer. Analyze the quiz question and generate Python code to find the correct answer.

QUIZ QUESTION/INSTRUCTIONS:
{instructions}

AVAILABLE DATA FILES:
{json.dumps(data_context, indent=2)}
{extracted_section}

MEDIA CONTENT:
{("\n".join(media_text) if media_text else "No media files.")}

{failure_context}
{strict_text_constraint}

CRITICAL CONSTRAINTS - ONLY USE THESE LIBRARIES:
- pandas (pd)
- numpy (np)
- json
- base64
- re (regex)
- math
- statistics
- datetime
- collections

DO NOT USE:
- sklearn, tensorflow, pytorch, or any ML libraries
- visualization libraries
- PDF/image-specific external libraries
- Any external libraries not listed above

Guidance:
- Parse and compute using ONLY the provided text snippets and data files.
- Use regex/string operations over the extracted text to find the required values.
- If numeric rounding is requested, apply it accordingly.
- Return only the answer value in variable 'answer'. If insufficient info, set 'answer' to a descriptive error string.

Respond with a JSON object:
{{
  "reasoning": "Brief explanation of your approach",
  "answer_type": "number|string|boolean|dict|dataframe",
  "code": "# Python code here\\nimport pandas as pd\\nimport numpy as np\\nimport re\\nimport json\\n\\n# Your solution\\nanswer = ..."
}}
"""
        # ...existing code to call LLM and parse JSON...
        try:
            system_prompt = "You are a Python programming expert. Generate clean, executable code. Return only valid JSON."
            
            headers = {
                "accept": "*/*",
                "authorization": f"Bearer {AIPIPE_TOKEN}",
                "content-type": "application/json",
            }
            
            # Build messages
            if image_contents:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + image_contents
                    }
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "temperature": 0.1
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content'].strip()
                else:
                    return {
                        'success': False,
                        'error': 'Invalid response from LLM'
                    }
            
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Sanitize illegal backslash escapes (e.g. \s, \n inside plain text) before json.loads
            def _sanitize_json(raw: str) -> str:
                # Normalize CRLF
                raw = raw.replace('\r\n', '\n')
                # Escape lone backslashes not part of valid JSON escapes
                raw = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', raw)
                # Remove trailing commas in objects/arrays
                raw = re.sub(r',\s*([}\]])', r'\1', raw)
                return raw

            sanitized = _sanitize_json(content)

            # Try primary parse, else robust fallback to extract the largest JSON object
            response_data = None
            try:
                response_data = json.loads(sanitized)
            except Exception as e:
                logger.error(f"Primary JSON parse failed: {e}")
                # Fallback: find the outermost JSON object by brace matching
                try:
                    # Find first '{' and last '}' to capture object region
                    first = sanitized.find('{')
                    last = sanitized.rfind('}')
                    if first != -1 and last != -1 and last > first:
                        candidate = sanitized[first:last+1]
                        candidate = _sanitize_json(candidate)
                        response_data = json.loads(candidate)
                    else:
                        # Try to extract via regex in case of additional text
                        brace_match = re.search(r'\{.*\}', sanitized, re.DOTALL)
                        if brace_match:
                            candidate = _sanitize_json(brace_match.group(0))
                            response_data = json.loads(candidate)
                except Exception as e2:
                    return {
                        'success': False,
                        'error': f'Code generation failed: JSON parse error ({e2})'
                    }

            # Ensure we have required fields, provide defaults if missing
            code_str = (response_data or {}).get('code', '')
            reasoning = (response_data or {}).get('reasoning', '')
            answer_type = (response_data or {}).get('answer_type', 'auto')
            if not code_str or not isinstance(code_str, str):
                return {
                    'success': False,
                    'error': 'Code generation failed: missing code'
                }

            return {
                'success': True,
                'code': code_str,
                'reasoning': reasoning,
                'answer_type': answer_type
            }
        except Exception as e:
            # Catch any unexpected errors and return a consistent failure structure
            logger.error(f"Error generating solution code: {e}")
            return {
                'success': False,
                'error': f'Code generation failed: {str(e)}'
            }
    
    async def _execute_generated_code(
        self,
        code: str,
        data_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute the LLM-generated Python code safely"""
        try:
            # Validate Python syntax before execution
            try:
                ast.parse(code)
            except SyntaxError as e:
                logger.error(f"Code has syntax error: {e}")
                code = self._fix_common_syntax_errors(code)
                try:
                    ast.parse(code)
                except SyntaxError as e2:
                    logger.error(f"Could not fix syntax error: {e2}")
                    return {
                        'success': False,
                        'error': f'Code has syntax error: {str(e2)}'
                    }
            
            # Create execution environment
            exec_globals = {
                'pd': pd,
                'np': __import__('numpy'),
                'json': json,
                'base64': base64,
                'os': os,
                '__builtins__': __builtins__
            }
            
            # Add file paths to environment
            if isinstance(data_files, list):
                for idx, file_info in enumerate(data_files):
                    if isinstance(file_info, dict):
                        exec_globals[f'file_{idx}'] = file_info.get('processed_file')
                    else:
                        exec_globals[f'file_{idx}'] = file_info
            elif isinstance(data_files, dict):
                for idx, (key, file_info) in enumerate(data_files.items()):
                    exec_globals[f'file_{idx}'] = file_info.get('processed_file')
            
            # Capture output
            output_buffer = StringIO()
            error_buffer = StringIO()
            
            # Execute code
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals)
            
            # Get the answer
            if 'answer' not in exec_globals:
                return {
                    'success': False,
                    'error': 'Code did not produce an "answer" variable'
                }
            
            answer = exec_globals['answer']
            
            # Determine answer type
            if isinstance(answer, pd.DataFrame):
                answer_type = 'dataframe'
            elif isinstance(answer, (int, float)):
                answer_type = 'number'
            elif isinstance(answer, bool):
                answer_type = 'boolean'
            elif isinstance(answer, str):
                answer_type = 'string'
            elif isinstance(answer, (dict, list)):
                answer_type = 'auto'
            else:
                answer_type = 'auto'
            
            return {
                'success': True,
                'result': answer,
                'answer_type': answer_type,
                'stdout': output_buffer.getvalue(),
                'stderr': error_buffer.getvalue()
            }
            
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                'success': False,
                'error': f'Code execution failed: {str(e)}'
            }
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Attempt to fix common syntax errors in generated code"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix mismatched brackets
            if '(' in line and ')' not in line and ']' in line:
                line = line.replace(']', ')')
            elif '[' in line and ']' not in line and ')' in line:
                line = line.replace(')', ']')
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _format_and_validate_answer(
        self,
        answer: Any,
        answer_type: str
    ) -> Dict[str, Any]:
        """Format and validate final answer"""
        try:
            # Handle DataFrame -> base64 CSV
            if answer_type == 'dataframe' and isinstance(answer, pd.DataFrame):
                csv_buffer = StringIO()
                answer.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')
                
                # Compress if too large
                if len(csv_bytes) > 500_000:
                    import gzip
                    csv_bytes = gzip.compress(csv_bytes)
                
                value = base64.b64encode(csv_bytes).decode('utf-8')
                final_type = 'base64'
            
            # Convert to appropriate type
            elif answer_type == 'number':
                value = float(answer) if not isinstance(answer, (int, float)) else answer
                final_type = 'number'
            elif answer_type == 'string':
                value = str(answer)
                final_type = 'string'
            elif answer_type == 'boolean':
                value = bool(answer)
                final_type = 'boolean'
            elif answer_type == 'dict':
                value = dict(answer) if not isinstance(answer, dict) else answer
                final_type = 'dict'
            else:
                # Auto-detect
                if isinstance(answer, bool):
                    value = answer
                    final_type = 'boolean'
                elif isinstance(answer, (int, float)):
                    value = answer
                    final_type = 'number'
                elif isinstance(answer, dict):
                    value = answer
                    final_type = 'dict'
                elif isinstance(answer, list):
                    value = {'data': answer}
                    final_type = 'dict'
                else:
                    value = str(answer)
                    final_type = 'string'
            
            # Validate JSON size
            json_str = json.dumps(value)
            if len(json_str) > self.max_json_size:
                if isinstance(value, (list, dict)) and final_type != 'base64':
                    value = {'error': 'Result too large', 'sample': str(value)[:1000]}
            
            return {
                'value': value,
                'type': final_type
            }
        
        except Exception as e:
            logger.error(f"Error formatting answer: {e}")
            return {
                'value': str(answer),
                'type': 'string'
            }


# Module-level function for external use
async def generate_quiz_answer(
    instructions: str,
    processed_data: list,
    page_html: str = "",
    media_files: list = None,
    image_text_extractions: list = None,
    failure_reason: str = None,
    structured_data: list = None
) -> dict:
    """Generate answer for quiz question using LLM"""
    generator = AnswerGenerator()
    return await generator.generate_answer(
        instructions=instructions,
        processed_data=processed_data,
        page_html=page_html,
        media_files=media_files,
        image_text_extractions=image_text_extractions,
        failure_reason=failure_reason,
        structured_data=structured_data  # pass through for fast-path ML
    )
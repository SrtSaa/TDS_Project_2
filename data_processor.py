import os
import httpx
import json
import logging
import asyncio
import pandas as pd
import io
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import time
import mimetypes
import shutil
import zipfile
import tarfile
import py7zr
import re
import zlib

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

# Constants
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DOWNLOAD_TIMEOUT = 30  # seconds
LLM_TIMEOUT = 180  # seconds - increased for large files

# Supported file types
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff', '.ico'}
COMPRESSED_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tgz'}
# Remove .pdf from binary so we can read and parse text from it
BINARY_EXTENSIONS = AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | {'.exe', '.dll', '.bin'}
TEXT_EXTENSIONS = {'.txt', '.csv', '.json', '.xml', '.html', '.md', '.log', '.tsv'}

# LLM API Configuration
LLM_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "authorization": f"Bearer {AIPIPE_TOKEN}",
    "content-type": "application/json",
}


def ensure_directories():
    """Create necessary directories if they don't exist"""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directories exist: {RAW_DATA_DIR}, {PROCESSED_DATA_DIR}")


def get_file_extension(url: str, content_type: Optional[str] = None) -> str:
    """
    Determine file extension from URL or Content-Type header
    
    Args:
        url: File URL
        content_type: Content-Type header value
        
    Returns:
        File extension with dot (e.g., '.csv')
    """
    # First try to get extension from URL
    url_path = url.split('?')[0]  # Remove query parameters
    url_ext = Path(url_path).suffix.lower()
    
    if url_ext:
        return url_ext
    
    # If no extension in URL, try Content-Type
    if content_type:
        # Handle common MIME types
        mime_to_ext = {
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/flac': '.flac',
            'audio/ogg': '.ogg',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/svg+xml': '.svg',
        }
        
        content_type_base = content_type.split(';')[0].strip().lower()
        if content_type_base in mime_to_ext:
            return mime_to_ext[content_type_base]
        
        ext = mimetypes.guess_extension(content_type_base)
        if ext:
            return ext
    
    # Default to .txt if cannot determine
    return '.txt'


def is_binary_file(extension: str) -> bool:
    """Check if file extension is binary (audio, image, etc.)"""
    return extension.lower() in BINARY_EXTENSIONS


def is_audio_file(extension: str) -> bool:
    """Check if file extension is audio"""
    return extension.lower() in AUDIO_EXTENSIONS


def is_image_file(extension: str) -> bool:
    """Check if file extension is image"""
    return extension.lower() in IMAGE_EXTENSIONS


def is_compressed_file(extension: str) -> bool:
    """Check if file extension is a compressed archive"""
    return extension.lower() in COMPRESSED_EXTENSIONS


def extract_binary_metadata(filepath: str, extension: str) -> Dict:
    """
    Extract basic metadata from binary files (audio/image)
    
    Args:
        filepath: Path to file
        extension: File extension
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        'file_type': 'unknown',
        'size_bytes': 0,
        'size_readable': ''
    }
    
    try:
        file_size = os.path.getsize(filepath)
        metadata['size_bytes'] = file_size
        
        # Human readable size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if file_size < 1024.0:
                metadata['size_readable'] = f"{file_size:.2f} {unit}"
                break
            file_size /= 1024.0
        
        # Determine file type
        if is_audio_file(extension):
            metadata['file_type'] = 'audio'
                
        elif is_image_file(extension):
            metadata['file_type'] = 'image'
            
            # Try to get image dimensions if possible
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    metadata['width'] = img.width
                    metadata['height'] = img.height
                    metadata['format'] = img.format
                    metadata['mode'] = img.mode
            except ImportError:
                pass
            except Exception:
                pass
        
        else:
            metadata['file_type'] = 'binary'
            
    except Exception as e:
        logger.error(f"Error extracting metadata from {filepath}: {e}")
    
    return metadata


async def extract_compressed_file(filepath: str, extension: str) -> Dict[str, any]:
    """
    Extract compressed files to raw directory
    
    Args:
        filepath: Path to compressed file
        extension: File extension
        
    Returns:
        Dictionary with extraction results
    """
    extract_dir = RAW_DATA_DIR / f"extracted_{Path(filepath).stem}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    
    try:
        logger.info(f"Extracting {filepath}")
        
        if extension in ['.zip']:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = [str(extract_dir / name) for name in zip_ref.namelist() if not name.endswith('/')]
        
        elif extension in ['.tar', '.gz', '.tgz', '.bz2', '.xz']:
            mode = 'r:*'  # Auto-detect compression
            with tarfile.open(filepath, mode) as tar_ref:
                tar_ref.extractall(extract_dir)
                extracted_files = [str(extract_dir / member.name) for member in tar_ref.getmembers() if member.isfile()]
        
        elif extension == '.7z':
            try:
                with py7zr.SevenZipFile(filepath, mode='r') as archive:
                    archive.extractall(path=extract_dir)
                    extracted_files = [str(extract_dir / name) for name in archive.getnames()]
            except Exception as e:
                logger.error(f"7z extraction failed: {e}")
                return {
                    'success': False,
                    'error': f'7z extraction requires py7zr: {str(e)}',
                    'extracted_files': []
                }
        
        elif extension == '.rar':
            try:
                import rarfile
                with rarfile.RarFile(filepath) as rar_ref:
                    rar_ref.extractall(extract_dir)
                    extracted_files = [str(extract_dir / name) for name in rar_ref.namelist() if not name.endswith('/')]
            except ImportError:
                logger.error("RAR extraction requires rarfile and unrar")
                return {
                    'success': False,
                    'error': 'RAR extraction requires rarfile package',
                    'extracted_files': []
                }
            except Exception as e:
                logger.error(f"RAR extraction failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'extracted_files': []
                }
        
        else:
            return {
                'success': False,
                'error': f'Unsupported compression format: {extension}',
                'extracted_files': []
            }
        
        logger.info(f"Extracted {len(extracted_files)} files")
        
        return {
            'success': True,
            'extracted_files': extracted_files,
            'extract_dir': str(extract_dir)
        }
    
    except Exception as e:
        logger.error(f"Error extracting {filepath}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'extracted_files': []
        }


def preprocess_directly(file_content: str, preprocessing_instructions: str, file_type: str) -> Dict:
    """
    Direct preprocessing without LLM for common operations
    
    Args:
        file_content: Raw file content
        preprocessing_instructions: Instructions
        file_type: File extension
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    try:
        import re
        
        instructions_lower = preprocessing_instructions.lower().strip()
        
        # Handle text paragraph extraction
        if file_type.lower() in ['.txt', '.text'] and 'paragraph' in instructions_lower and 'extract' in instructions_lower:
            # Extract number from instruction
            numbers = re.findall(r'\d+', instructions_lower)
            if numbers:
                n = int(numbers[0])
                
                # Split by blank lines to get paragraphs
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', file_content) if p.strip()]
                
                # Take first N paragraphs
                selected = paragraphs[:n]
                
                # Clean up each paragraph: normalize internal whitespace
                # Remove excessive blank lines within paragraphs, keep single newlines
                cleaned = []
                for para in selected:
                    # Replace multiple spaces with single space
                    para = re.sub(r' +', ' ', para)
                    # Replace multiple newlines with single newline
                    para = re.sub(r'\n+', '\n', para)
                    # Remove trailing/leading whitespace per line
                    lines = [line.strip() for line in para.split('\n') if line.strip()]
                    para = '\n'.join(lines)
                    cleaned.append(para)
                
                # Join paragraphs with double newline
                result = '\n\n'.join(cleaned)
                
                return {
                    'success': True,
                    'preprocessed_data': result,
                    'format': 'txt',
                    'steps_performed': [f'Extracted first {len(selected)} paragraphs from {len(paragraphs)} total'],
                    'needs_processing': True,
                    'validation': {
                        'paragraphs_found': len(paragraphs),
                        'paragraphs_extracted': len(selected),
                        'output_length': len(result)
                    }
                }
        
        # Handle CSV operations
        if file_type.lower() in ['.csv', '.tsv']:
            df = pd.read_csv(io.StringIO(file_content))
            initial_rows = len(df)
            initial_cols = len(df.columns)
            steps = []
            
            # Filter rows based on conditions
            if 'filter' in instructions_lower or 'year' in instructions_lower:
                # Look for year column (case-insensitive)
                year_col = None
                for col in df.columns:
                    if col.lower() == 'year':
                        year_col = col
                        break
                
                if year_col:
                    # Extract comparison pattern (e.g., ">= 2010", "year >= 2010")
                    filter_match = re.search(r'(>=|<=|>|<|=)\s*(\d{4})', instructions_lower)
                    if filter_match:
                        operator = filter_match.group(1)
                        value = int(filter_match.group(2))
                        
                        before = len(df)
                        if operator == '>=':
                            df = df[df[year_col] >= value]
                        elif operator == '<=':
                            df = df[df[year_col] <= value]
                        elif operator == '>':
                            df = df[df[year_col] > value]
                        elif operator == '<':
                            df = df[df[year_col] < value]
                        elif operator == '=':
                            df = df[df[year_col] == value]
                        
                        filtered = before - len(df)
                        steps.append(f"Filtered: kept {len(df)} rows where {year_col} {operator} {value} (removed {filtered} rows)")
                        logger.info(f"Filtered CSV: {before} rows -> {len(df)} rows (condition: {year_col} {operator} {value})")
            
            # Remove rows with missing values in specific column
            if 'remove' in instructions_lower and 'missing' in instructions_lower:
                # Find column name mentioned
                for col in df.columns:
                    if col.lower() in instructions_lower:
                        before = len(df)
                        df = df.dropna(subset=[col])
                        removed = before - len(df)
                        steps.append(f"Removed {removed} rows with missing {col}")
                        logger.info(f"Removed {removed} rows with missing {col}")
                        break
            
            # Standardize column names
            if 'standardize' in instructions_lower and 'column' in instructions_lower:
                if 'lowercase' in instructions_lower or 'lower' in instructions_lower:
                    df.columns = df.columns.str.lower()
                    steps.append(f"Standardized {len(df.columns)} column names to lowercase")
            
            # Remove/drop specific columns
            if ('remove' in instructions_lower or 'drop' in instructions_lower or 'omit' in instructions_lower) and 'column' in instructions_lower:
                # Find column names mentioned
                cols_to_drop = []
                for col in df.columns:
                    if col.lower() in instructions_lower:
                        cols_to_drop.append(col)
                
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    steps.append(f"Removed columns: {', '.join(cols_to_drop)}")
            
            if not steps:
                steps.append('No matching preprocessing rules found')
            
            result = df.to_csv(index=False)
            
            return {
                'success': True,
                'preprocessed_data': result,
                'format': 'csv',
                'steps_performed': steps,
                'needs_processing': len(steps) > 0 and steps[0] != 'No matching preprocessing rules found',
                'validation': {
                    'rows_before': initial_rows,
                    'rows_after': len(df),
                    'rows_removed': initial_rows - len(df),
                    'columns_before': initial_cols,
                    'columns_after': len(df.columns)
                }
            }
        
        # Handle JSON operations
        if file_type.lower() == '.json':
            data = json.loads(file_content)
            
            # If it's a list of records
            if isinstance(data, list):
                df = pd.DataFrame(data)
                initial_rows = len(df)
                steps = []
                
                # Extract specific fields
                if 'extract' in instructions_lower and 'field' in instructions_lower:
                    # Find field names mentioned
                    available_cols = df.columns.tolist()
                    
                    # Extract nested fields
                    fields_to_extract = []
                    for col in available_cols:
                        if col.lower() in instructions_lower:
                            fields_to_extract.append(col)
                    
                    # Handle nested fields (e.g., company.name)
                    nested_pattern = r'(\w+)\.(\w+)'
                    nested_matches = re.findall(nested_pattern, instructions_lower)
                    
                    for parent, child in nested_matches:
                        # Find matching column
                        for col in available_cols:
                            if col.lower() == parent.lower():
                                # Extract nested field
                                if df[col].dtype == 'object':
                                    try:
                                        df[f'{parent}_{child}'] = df[col].apply(
                                            lambda x: x.get(child) if isinstance(x, dict) else None
                                        )
                                        fields_to_extract.append(f'{parent}_{child}')
                                    except:
                                        pass
                    
                    if fields_to_extract:
                        df = df[fields_to_extract]
                        steps.append(f"Extracted fields: {', '.join(fields_to_extract)}")
                
                # Remove specific fields
                if 'remove' in instructions_lower and 'field' in instructions_lower:
                    # Check for nested field pattern
                    nested_pattern = r'(\w+)\.(\w+)'
                    nested_matches = re.findall(nested_pattern, instructions_lower)
                    
                    for parent, child in nested_matches:
                        for col in df.columns:
                            if col.lower() == parent.lower():
                                # Check if field exists in nested structure
                                if df[col].dtype == 'object':
                                    try:
                                        # Field doesn't exist in nested structure - no change needed
                                        sample = df[col].iloc[0] if len(df) > 0 else None
                                        if isinstance(sample, dict) and child not in sample:
                                            steps.append(f"Field {parent}.{child} not found - no changes made")
                                    except:
                                        pass
                        
                result = df.to_json(orient='records', indent=2)
                
                return {
                    'success': True,
                    'preprocessed_data': result,
                    'format': 'json',
                    'steps_performed': steps if steps else ['No changes made - conditions not met'],
                    'needs_processing': len(steps) > 0,
                    'validation': {
                        'rows_before': initial_rows,
                        'rows_after': len(df),
                        'columns_processed': len(df.columns)
                    }
                }
        
        # If no handler matched, return original
        return {
            'success': False,
            'preprocessed_data': file_content,
            'format': file_type.replace('.', ''),
            'steps_performed': ['Direct preprocessing not applicable'],
            'needs_processing': False
        }
        
    except Exception as e:
        logger.error(f"Direct preprocessing failed: {e}", exc_info=True)
        return {
            'success': False,
            'preprocessed_data': file_content,
            'format': file_type.replace('.', ''),
            'steps_performed': [f'Error: {str(e)}'],
            'needs_processing': False
        }


async def preprocess_with_llm(file_content: str, preprocessing_instructions: str, filename: str, file_type: str, failure_reason: Optional[str] = None) -> Dict:
    """
    Use two-stage LLM approach to preprocess data with maximum accuracy
    
    Stage 1: Understand instructions and generate preprocessing plan
    Stage 2: Execute preprocessing with pandas code
    
    Args:
        file_content: Raw file content
        preprocessing_instructions: Instructions from quiz
        filename: Original filename
        file_type: File extension
        failure_reason: Previous failure reason for retry attempts
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    if failure_reason:
        logger.warning(f"Retry attempt - Previous failure: {failure_reason}")
    
    # Check if this is a binary file
    if is_binary_file(file_type):
        # Parse metadata if available
        try:
            metadata = json.loads(file_content)
            file_category = metadata.get('file_type', 'binary')
        except:
            file_category = 'binary'
        
        # Check if preprocessing is needed
        if not should_preprocess_file(file_type, preprocessing_instructions):
            return {
                'success': True,
                'preprocessed_data': file_content,
                'format': file_type.replace('.', ''),
                'steps_performed': [f'No preprocessing needed for {file_category} file - copied to processed directory'],
                'needs_processing': False,
                'is_binary': True,
                'copy_binary': True,  # Flag to indicate binary copy is needed
                'validation': metadata if isinstance(metadata, dict) else {}
            }
        
        # If preprocessing instructions mention audio/image operations
        return {
            'success': True,
            'preprocessed_data': file_content,
            'format': file_type.replace('.', ''),
            'steps_performed': [
                f'{file_category.capitalize()} file detected',
                'Note: Binary file preprocessing requires specialized tools',
                f'Instructions: {preprocessing_instructions[:100]}...'
            ],
            'needs_processing': False,
            'is_binary': True,
            'copy_binary': True,  # Still copy the binary file
            'validation': metadata if isinstance(metadata, dict) else {}
        }
    
    # Try direct preprocessing first (faster and more reliable)
    direct_result = preprocess_directly(file_content, preprocessing_instructions, file_type)
    if direct_result['success']:
        return direct_result
    
    # Get data structure info
    lines = file_content.split('\n')
    header_line = lines[0] if lines else ""
    # For text files, show actual beginning to give LLM context
    sample_rows = '\n'.join(lines[:3]) if len(lines) > 3 else file_content[:500]
    
    # STAGE 1: Parse and understand instructions
    failure_context = f"\n\nPREVIOUS FAILURE: {failure_reason}\nPlease correct the approach to avoid this error." if failure_reason else ""
    
    stage1_system_prompt = """You are an expert data analyst specializing in understanding data preprocessing requirements.

Your task is to analyze ANY preprocessing instruction and convert it into precise, executable actions.

CRITICAL RULES:
1. Read the instruction CAREFULLY and interpret EXACTLY what it asks for
2. If the instruction asks to remove/delete a field that doesn't exist, the operation should be NO-OP (do nothing)
3. Consider ALL possible phrasings and synonyms
4. Do NOT assume or add requirements not stated
5. When processing data, ALWAYS process ALL records - NEVER truncate or sample
6. Handle nested structures (e.g., company.business means the 'business' field inside 'company' object)

Return JSON:
{
    "understood_requirements": ["precise list of what needs to be done"],
    "fields_mentioned": ["exact field names or nested paths like 'company.business'"],
    "operations": [
        {
            "type": "extract|remove|filter|select|transform|rename|etc",
            "target": "field name or nested path",
            "condition": "exact condition",
            "if_exists": true
        }
    ],
    "should_process_all_records": true
}"""

    stage1_payload = {
        "model": "openai/gpt-4o",
        "max_tokens": 2000,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": stage1_system_prompt},
            {"role": "user", "content": f"""Analyze this preprocessing instruction:

INSTRUCTION: {preprocessing_instructions.strip() or "No changes needed. Return data as-is."}{failure_context}

DATA STRUCTURE:
- File type: {file_type}
- Sample content:
{sample_rows}

Determine what needs to be done. If a field mentioned doesn't exist, note that no changes should be made."""}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(LLM_URL, headers=LLM_HEADERS, json=stage1_payload)
            response.raise_for_status()
            requirements = json.loads(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}", exc_info=True)
        return {
            'success': False,
            'preprocessed_data': file_content,
            'format': file_type.replace('.', ''),
            'steps_performed': ['Failed to understand instructions'],
            'needs_processing': False
        }
    
    # STAGE 2: Generate and execute code
    # Determine the correct pandas read function based on file type
    if file_type.lower() in ['.json']:
        read_function = "pd.read_json(io.StringIO(input_data))"
    elif file_type.lower() in ['.csv', '.tsv']:
        read_function = "pd.read_csv(io.StringIO(input_data))"
    else:
        read_function = "pd.read_csv(io.StringIO(input_data))"
    
    stage2_system_prompt = f"""You are an expert Python pandas developer. Generate PERFECT, EXECUTABLE code for data preprocessing.

ABSOLUTE REQUIREMENTS:
1. The variable 'input_data' is already defined and contains the FULL actual file content as a string
2. For TEXT files extracting paragraphs:
   - A paragraph is text separated by one or more blank lines (lines with only whitespace)
   - Use re.split(r'\\n\\s*\\n', input_data) to split by blank lines
   - Each resulting block (even if it contains internal line breaks) is ONE paragraph
   - Filter empty strings: [p.strip() for p in ... if p.strip()]
   - Take EXACTLY the number requested (e.g., first 5 means paragraphs[0:5])
   - Count carefully - if file has fewer paragraphs than requested, take all available
   - Print: total found, requested, and actually extracted
3. NEVER create sample/fake data
4. Code MUST create variable 'df' with the result

CODE TEMPLATE:
```python
import pandas as pd
import re

# Split by blank lines (one or more)
paragraphs = [p.strip() for p in re.split(r'\\n\\s*\\n', input_data) if p.strip()]
total_found = len(paragraphs)
print(f"Total paragraphs found: {{total_found}}")

# Extract exactly N (or all if fewer available)
N = [NUMBER]  # e.g., 5
selected = paragraphs[:N]
actual_extracted = len(selected)
print(f"Requested: {{N}}, Actually extracted: {{actual_extracted}}")

# Join back with double newline
result_text = '\\n\\n'.join(selected)
df = pd.DataFrame({{'text': [result_text]}})
print(f"Result: {{actual_extracted}} paragraphs, {{len(result_text)}} characters")
```

Return JSON:
{{
    "python_code": "code string",
    "explanation": "description"
}}"""

    stage2_payload = {
        "model": "openai/gpt-4o",
        "max_tokens": 2000,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": stage2_system_prompt},
            {"role": "user", "content": f"""Generate code for: {preprocessing_instructions.strip()}

File type: {file_type}
File begins with: "{sample_rows[:150]}"

CRITICAL:
- 'input_data' variable already exists with full file
- Split by blank lines: re.split(r'\\n\\s*\\n', input_data)
- Each block = 1 paragraph (even if it has internal line breaks)
- Extract EXACTLY the number requested in instruction
- If instruction says "first 5", use paragraphs[:5]
- Print counts: total found, requested, extracted

Generate code now."""}
        ],
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(MAX_RETRIES):
        python_code = ""  # Initialize to avoid UnboundLocalError
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                response = await client.post(LLM_URL, headers=LLM_HEADERS, json=stage2_payload)
                response.raise_for_status()
                
                # Get raw response content
                raw_content = response.json()['choices'][0]['message']['content']
                
                # Try to parse JSON with better error handling
                try:
                    code_response = json.loads(raw_content)
                    python_code = code_response.get('python_code', '')
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error: {json_err}")
                    
                    # Try to extract from code blocks
                    import re
                    code_match = re.search(r'```python\n(.*?)\n```', raw_content, re.DOTALL)
                    if code_match:
                        python_code = code_match.group(1)
                    else:
                        raise ValueError(f"Could not parse JSON response: {json_err}")
                
                if not python_code:
                    raise ValueError("No code generated")

                # Sanitize disallowed imports (avoid PDF libraries that are not available)
                import re as _re
                disallowed = [
                    r'^\s*import\s+PyPDF2\b.*',
                    r'^\s*from\s+PyPDF2\b.*',
                    r'^\s*import\s+pdfplumber\b.*',
                    r'^\s*from\s+pdfplumber\b.*',
                    r'^\s*import\s+fitz\b.*',          # PyMuPDF
                    r'^\s*from\s+fitz\b.*',
                ]
                for pat in disallowed:
                    python_code = '\n'.join(line for line in python_code.splitlines() if not _re.match(pat, line))

                # Execute code
                local_vars = {'pd': pd, 'io': io, 'json': json, 're': __import__('re'), 'input_data': file_content}
                
                # Capture output
                import sys
                from io import StringIO as IOStringIO
                old_stdout = sys.stdout
                sys.stdout = captured = IOStringIO()
                
                try:
                    exec(python_code, local_vars)
                    execution_log = captured.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                if 'df' not in local_vars:
                    raise ValueError("Code didn't create 'df' variable")
                
                processed_df = local_vars['df']
                rows_before = len(lines)
                rows_after = len(processed_df)
                
                # Determine output format based on original file type
                if file_type.lower() == '.json':
                    output_data = processed_df.to_json(orient='records', indent=2)
                    output_format = 'json'
                elif file_type.lower() in ['.txt', '.text']:
                    # For text files, output as text not CSV
                    output_data = processed_df.iloc[0, 0] if len(processed_df) > 0 and len(processed_df.columns) > 0 else str(processed_df)
                    output_format = 'txt'
                else:
                    output_data = processed_df.to_csv(index=False)
                    output_format = 'csv'
                
                return {
                    'success': True,
                    'preprocessed_data': output_data,
                    'format': output_format,
                    'steps_performed': requirements.get('understood_requirements', []),
                    'needs_processing': True,
                    'validation': {
                        'rows_before': rows_before,
                        'rows_after': rows_after,
                        'rows_removed': rows_before - rows_after,
                        'columns_processed': len(processed_df.columns),
                        'execution_log': execution_log
                    }
                }
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                # Only add previous code if it was successfully extracted
                if python_code:
                    stage2_payload["messages"].append({
                        "role": "assistant",
                        "content": json.dumps({"python_code": python_code, "explanation": "Previous attempt"})
                    })
                stage2_payload["messages"].append({
                    "role": "user",
                    "content": f"""Previous attempt failed: {str(e)}

Fix the code:
- Use re.split(r'\\n\\s*\\n', input_data) to split by blank lines
- Take EXACTLY first N paragraphs as specified: paragraphs[:N]
- Print counts for verification
- File type: {file_type}"""
                })
                await asyncio.sleep(RETRY_DELAY)
                continue
    
    # Final fallback
    logger.error("All attempts failed, returning original data")
    return {
        'success': False,
        'preprocessed_data': file_content,
        'format': file_type.replace('.', ''),
        'steps_performed': ['Preprocessing failed after all retries'],
        'needs_processing': False
    }


def save_processed_file(data: str, original_filename: str, output_format: str) -> str:
    """
    Save preprocessed data to processed directory
    
    Args:
        data: Preprocessed data
        original_filename: Original filename
        output_format: Output format (csv, json, txt, etc.)
        
    Returns:
        Path to saved file
    """
    # Create output filename
    base_name = Path(original_filename).stem
    extension = f".{output_format}" if not output_format.startswith('.') else output_format
    output_filename = f"{base_name}_processed{extension}"
    output_path = PROCESSED_DATA_DIR / output_filename
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(data)
    
    return str(output_path)


def copy_binary_to_processed(source_filepath: str) -> str:
    """
    Copy binary file (audio/image) to processed directory without modification
    
    Args:
        source_filepath: Path to source binary file
        
    Returns:
        Path to copied file in processed directory
    """
    source_path = Path(source_filepath)
    output_filename = f"{source_path.stem}_processed{source_path.suffix}"
    output_path = PROCESSED_DATA_DIR / output_filename
    
    # Copy file
    shutil.copy2(source_filepath, output_path)
    
    return str(output_path)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Minimal PDF text extractor (handles FlateDecode streams, BT/ET blocks, and Tj/TJ text operators)."""
    text_parts = []

    data_str = pdf_bytes.decode('latin-1', errors='ignore')

    # Extract and decompress FlateDecode streams
    stream_pattern = re.compile(r'<<[^>]*>>\s*stream\r?\n(.*?)\r?\nendstream', re.DOTALL)
    for match in stream_pattern.finditer(data_str):
        header = match.group(0).split('stream')[0]
        stream_data_raw = match.group(1).encode('latin-1', errors='ignore')
        if '/Filter' in header and '/FlateDecode' in header:
            try:
                decompressed = zlib.decompress(stream_data_raw)
                decom_str = decompressed.decode('latin-1', errors='ignore')
                # Extract text in parentheses inside streams
                for s in re.findall(r'\((.*?)\)', decom_str, re.DOTALL):
                    cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                    cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
                    if cleaned.strip():
                        text_parts.append(cleaned.strip())
                # Extract arrays used with TJ operator: [ ... ] TJ
                for arr in re.findall(r'\[(.*?)\]\s*TJ', decom_str, re.DOTALL):
                    for s in re.findall(r'\((.*?)\)', arr, re.DOTALL):
                        cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                        cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
                        if cleaned.strip():
                            text_parts.append(cleaned.strip())
            except Exception:
                # Non-decompressible: still try to extract parentheses
                for s in re.findall(r'\((.*?)\)', match.group(1), re.DOTALL):
                    cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                    cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
                    if cleaned.strip():
                        text_parts.append(cleaned.strip())
        else:
            # Non-compressed stream: still try to extract parentheses
            for s in re.findall(r'\((.*?)\)', match.group(1), re.DOTALL):
                cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
                if cleaned.strip():
                    text_parts.append(cleaned.strip())

    # Extract text between BT ... ET blocks
    for bt_block in re.findall(r'BT(.*?)ET', data_str, re.DOTALL):
        # Text shown via Tj operator: (text) Tj
        for s in re.findall(r'\((.*?)\)\s*Tj', bt_block, re.DOTALL):
            cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
            cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
            if cleaned.strip():
                text_parts.append(cleaned.strip())
        # Arrays with TJ operator: [ (a) (b) ] TJ
        for arr in re.findall(r'\[(.*?)\]\s*TJ', bt_block, re.DOTALL):
            for s in re.findall(r'\((.*?)\)', arr, re.DOTALL):
                cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
                cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
                if cleaned.strip():
                    text_parts.append(cleaned.strip())
        # Fallback: any parentheses inside BT/ET
        for s in re.findall(r'\((.*?)\)', bt_block, re.DOTALL):
            cleaned = s.replace(r'\(', '(').replace(r'\)', ')').replace(r'\\', '\\')
            cleaned = re.sub(r'[\x00-\x08\x0B-\x1F]', '', cleaned)
            if cleaned.strip():
                text_parts.append(cleaned.strip())

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for t in text_parts:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    # Filter generic PDF boilerplate tokens
    def is_boilerplate(line: str) -> bool:
        l = line.strip()
        if not l:
            return True
        # Common header/footer markers
        if l.startswith('%PDF-'):
            return True
        if l.upper() in {'%PDF', 'EOF'}:
            return True
        # Pure object markers or numeric-only noise
        if re.fullmatch(r'\d+\s+\d+\s+obj', l, re.IGNORECASE):
            return True
        if re.fullmatch(r'\d+', l):
            return True
        return False

    filtered = [t for t in ordered if not is_boilerplate(t)]
    combined = "\n".join(filtered)
    return combined.strip()


def extract_pdf_text(filepath: str) -> Dict[str, Any]:
    """
    Robust PDF text extraction with layered fallbacks:
    1. pdfminer.six (layout-aware)
    2. PyPDF2 (simple per-page extraction)
    3. Minimal raw stream/BT/ET parser (_extract_pdf_text)
    Returns dict with keys: text (str), method (str), errors (list)
    """
    errors = []
    text = ""
    method = ""

    # Attempt pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
        text_pm = pdf_extract_text(filepath) or ""
        cleaned_pm = text_pm.strip()
        if cleaned_pm:
            text = cleaned_pm
            method = "pdfminer.six"
            return {"text": text, "method": method, "errors": errors}
        else:
            errors.append("pdfminer produced empty text")
    except Exception as e:
        errors.append(f"pdfminer error: {e}")

    # Attempt PyPDF2
    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = []
            for page in reader.pages:
                try:
                    ptxt = page.extract_text() or ""
                    if ptxt.strip():
                        parts.append(ptxt)
                except Exception:
                    continue
            merged = "\n".join(p.strip() for p in parts if p.strip()).strip()
            if merged:
                text = merged
                method = "PyPDF2"
                return {"text": text, "method": method, "errors": errors}
            else:
                errors.append("PyPDF2 produced empty text")
    except Exception as e:
        errors.append(f"PyPDF2 error: {e}")

    # Final fallback: raw stream parser
    try:
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()
        raw_text = _extract_pdf_text(pdf_bytes)
        if raw_text.strip():
            text = raw_text.strip()
            method = "raw_fallback"
        else:
            errors.append("raw fallback produced empty text")
    except Exception as e:
        errors.append(f"raw fallback error: {e}")

    if not text:
        text = "[PDF text extraction failed: " + "; ".join(errors) + "]"
        method = "failure"
    return {"text": text, "method": method, "errors": errors}


def read_file_content(filepath: str, extension: str) -> str:
    """
    Read file content as string
    """
    if extension.lower() == '.pdf':
        result = extract_pdf_text(filepath)
        return result["text"]

    # For binary files (audio/image), return metadata instead
    if is_binary_file(extension):
        metadata = extract_binary_metadata(filepath, extension)
        return json.dumps(metadata, indent=2)
    
    # For compressed files, return metadata
    if is_compressed_file(extension):
        metadata = {
            'file_type': 'compressed',
            'extension': extension,
            'size_bytes': os.path.getsize(filepath),
            'note': 'This is a compressed archive that will be extracted'
        }
        return json.dumps(metadata, indent=2)
    
    try:
        # Try UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # Try with latin-1 if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
            return content
        except Exception:
            metadata = {
                'file_type': 'unknown_binary',
                'extension': extension,
                'size_bytes': os.path.getsize(filepath),
                'note': 'File could not be read as text'
            }
            return json.dumps(metadata, indent=2)


def should_preprocess_file(extension: str, preprocessing_instructions: str) -> bool:
    """
    Determine if preprocessing should be applied to a file
    
    Args:
        extension: File extension
        preprocessing_instructions: Preprocessing instructions
        
    Returns:
        True if preprocessing should be applied
    """
    # If no instructions, skip preprocessing
    if not preprocessing_instructions or preprocessing_instructions.strip() == "":
        return False
    
    # Compressed files - always extract if they exist
    if is_compressed_file(extension):
        return True
    
    # For binary files, only preprocess if explicitly mentioned
    if is_binary_file(extension):
        instructions_lower = preprocessing_instructions.lower()
        
        # Check if audio/image processing is mentioned
        audio_keywords = ['audio', 'sound', 'music', 'transcript', 'speech']
        image_keywords = ['image', 'picture', 'photo', 'resize', 'crop', 'filter', 'convert']
        
        if is_audio_file(extension):
            return any(keyword in instructions_lower for keyword in audio_keywords)
        
        if is_image_file(extension):
            return any(keyword in instructions_lower for keyword in image_keywords)
        
        # Other binary files - no preprocessing
        return False
    
    # Text-based files - always preprocess if instructions exist
    return True


async def process_single_file(
    filepath: str,
    filename: str,
    extension: str,
    preprocessing_instructions: str,
    url: str = "",
    failure_reason: Optional[str] = None
) -> Dict:
    """
    Process a single file (helper function for code reuse)
    
    Args:
        filepath: Path to file
        filename: Filename
        extension: File extension
        preprocessing_instructions: Preprocessing instructions
        url: Original URL
        failure_reason: Previous failure reason for retry
        
    Returns:
        Processed file info dictionary
    """
    # Read file content
    file_content = read_file_content(filepath, extension)

    if extension.lower() == '.pdf':
        pdf_result = extract_pdf_text(filepath)
        output_path = save_processed_file(pdf_result["text"], filename, 'txt')
        return {
            'original_file': filepath,
            'processed_file': output_path,
            'original_url': url,
            'steps_performed': [f'Extracted full text from PDF using {pdf_result["method"]}'],
            'needs_processing': False,
            'validation': {
                'output_length': len(pdf_result["text"]),
                'extraction_method': pdf_result["method"],
                'errors': pdf_result["errors"]
            },
            'type': 'text_extracted'
        }

    # Preprocess with hybrid approach (direct + LLM fallback)
    preprocess_result = await preprocess_with_llm(
        file_content,
        preprocessing_instructions,
        filename,
        extension,
        failure_reason=failure_reason
    )

    # Handle binary file copying
    if preprocess_result.get('copy_binary') and preprocess_result.get('is_binary'):
        output_path = copy_binary_to_processed(filepath)
        return {
            'original_file': filepath,
            'processed_file': output_path,
            'original_url': url,
            'steps_performed': preprocess_result.get('steps_performed', []),
            'needs_processing': False,
            'is_binary': True,
            'validation': preprocess_result.get('validation', {})
        }
    elif preprocess_result.get('preprocessed_data'):
        output_path = save_processed_file(
            preprocess_result['preprocessed_data'],
            filename,
            preprocess_result.get('format', extension.replace('.', ''))
        )
        return {
            'original_file': filepath,
            'processed_file': output_path,
            'original_url': url,
            'steps_performed': preprocess_result.get('steps_performed', []),
            'needs_processing': preprocess_result.get('needs_processing', True),
            'validation': preprocess_result.get('validation', {}),
            'type': 'text' if extension.lower() in ('.txt','.csv','.json','.xml','.md','.log','.tsv') else 'processed'
        }
    else:
        raise Exception(f"No preprocessed data returned for {filename}")


async def download_file(url: str, index: int) -> Dict:
    """
    Download a file from URL
    
    Args:
        url: File URL to download
        index: File index for naming
        
    Returns:
        Dictionary with download results
    """
    try:
        # Determine extension from URL
        extension = get_file_extension(url)
        
        # Create filename
        filename = f"file_{index}{extension}"
        filepath = RAW_DATA_DIR / filename
        
        logger.info(f"Downloading {url}")
        
        # Try original URL first
        try:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Get content type from response
                content_type = response.headers.get('content-type', '')
                
                # Re-determine extension based on content-type if needed
                if extension == '.txt' and content_type:
                    extension = get_file_extension(url, content_type)
                    filename = f"file_{index}{extension}"
                    filepath = RAW_DATA_DIR / filename
                
                # Save file
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded {filename} ({len(response.content)} bytes)")
                
                return {
                    'success': True,
                    'filepath': str(filepath),
                    'filename': filename,
                    'extension': extension,
                    'url': url,
                    'size': len(response.content)
                }
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Try alternate paths for 404
                logger.warning(f"Got 404 for {url}, trying alternate paths...")
                
                # Extract filename from URL and try common patterns
                from urllib.parse import urlparse
                parsed = urlparse(url)
                path_parts = parsed.path.split('/')
                file_name = path_parts[-1] if path_parts else None
                
                if file_name:
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    alternate_urls = [
                        f"{base_url}/data/{file_name}",
                        f"{base_url}/files/{file_name}",
                        f"{base_url}/assets/{file_name}",
                        f"{base_url}/{file_name}",
                    ]
                    
                    for alt_url in alternate_urls:
                        if alt_url == url:
                            continue
                        
                        try:
                            logger.info(f"Trying alternate URL: {alt_url}")
                            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
                                response = await client.get(alt_url)
                                response.raise_for_status()
                                
                                with open(filepath, 'wb') as f:
                                    f.write(response.content)
                                
                                logger.info(f"Downloaded from alternate URL: {alt_url}")
                                
                                return {
                                    'success': True,
                                    'filepath': str(filepath),
                                    'filename': filename,
                                    'extension': extension,
                                    'url': alt_url,
                                    'size': len(response.content)
                                }
                        except:
                            continue
                
                # All alternates failed
                logger.error(f"All download attempts failed for {url}")
                raise e
            else:
                raise e
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return {
            'success': False,
            'error': str(e),
            'url': url
        }


async def process_all_files(
    file_links: List[str],
    preprocessing_instructions: str = "",
    failure_reason: Optional[str] = None
) -> Dict:
    """
    Main function to download and process all files
    
    Args:
        file_links: List of file URLs
        preprocessing_instructions: Text containing preprocessing instructions
        failure_reason: Previous failure reason for retry attempts
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting data processing for {len(file_links)} files")
    
    if failure_reason:
        logger.warning(f"Retry mode - Previous failure: {failure_reason}")
    
    # Ensure directories exist
    ensure_directories()
    
    results = {
        'success': True,
        'downloaded_files': [],
        'processed_files': [],
        'errors': []
    }
    
    # Step 1: Download all files
    for i, url in enumerate(file_links):
        download_result = await download_file(url, i)
        results['downloaded_files'].append(download_result)
        
        if not download_result.get('success'):
            results['errors'].append(f"Failed to download {url}")
            continue
    
    # Step 2: Process each downloaded file
    for download_info in results['downloaded_files']:
        if not download_info.get('success'):
            continue
        
        try:
            filepath = download_info['filepath']
            filename = download_info['filename']
            extension = download_info['extension']
            url = download_info.get('url', '')
            
            # Check if this is a compressed file
            if is_compressed_file(extension):
                # Extract the compressed file
                extract_result = await extract_compressed_file(filepath, extension)
                
                if not extract_result['success']:
                    results['errors'].append(f"Failed to extract {filename}: {extract_result.get('error')}")
                    continue
                
                # Process each extracted file
                for extracted_file in extract_result['extracted_files']:
                    try:
                        extracted_path = Path(extracted_file)
                        extracted_name = extracted_path.name
                        extracted_ext = extracted_path.suffix
                        
                        processed_info = await process_single_file(
                            filepath=str(extracted_path),
                            filename=extracted_name,
                            extension=extracted_ext,
                            preprocessing_instructions=preprocessing_instructions,
                            url=f"{url} (extracted from {filename})",
                            failure_reason=failure_reason
                        )
                        
                        results['processed_files'].append(processed_info)
                    
                    except Exception as e:
                        logger.error(f"Error processing extracted file {extracted_file}: {e}")
                        results['errors'].append(f"Processing error for extracted file {extracted_path.name}: {str(e)}")
            
            else:
                # Regular file processing
                processed_info = await process_single_file(
                    filepath=filepath,
                    filename=filename,
                    extension=extension,
                    preprocessing_instructions=preprocessing_instructions,
                    url=url,
                    failure_reason=failure_reason
                )
                
                results['processed_files'].append(processed_info)
                
        except Exception as e:
            logger.error(f"Error processing {download_info.get('filename', 'unknown')}: {e}")
            results['errors'].append(f"Processing error for {download_info.get('filename', 'unknown')}: {str(e)}")
    
    # Update overall success status
    if results['errors']:
        results['success'] = len(results['processed_files']) > 0
    
    logger.info(f"Processed {len(results['processed_files'])} files with {len(results['errors'])} errors")
    
    return results

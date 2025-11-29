from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import os
from typing import Optional
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path  # added for Path usage when mapping local image paths
import asyncio  # added for timeouts

# Import quiz fetcher module
from quiz_fetcher import fetch_and_render_quiz_page, cleanup
from data_processor import process_all_files
from answer_generator import generate_quiz_answer
from submission_handler import submit_answer, submit_dummy_answer, should_retry

# Import image processing utilities
from image_processor import extract_text_from_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")
    await cleanup()


# Initialize FastAPI app
app = FastAPI(title="Automated Quiz Solver API", lifespan=lifespan)

# Environment variable for secret validation
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")


# Pydantic model for request validation
class QuizSolveRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email: str = Field(..., description="User email address")
    secret: str = Field(..., description="Secret key for authentication")
    url: str = Field(..., description="Quiz URL to process")



# Custom exception handlers for malformed JSON and validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return 400"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid request body", "details": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle JSON decode errors and other malformed requests"""
    if "JSON" in str(exc) or "json" in str(type(exc).__name__.lower()):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Malformed JSON request"}
        )
    # Re-raise if it's not a JSON-related error
    raise exc


# Background task function
async def run_quiz_process(url: str, email: str, secret: str):
    """
    Background task to process quiz with multi-level support
    
    Args:
        url: Initial quiz URL to process
        email: User email for submission
        secret: Secret key for submission
    """
    import time
    
    logger.info(f"Starting quiz processing for {url}")
    
    current_url = url
    level = 1
    
    def infer_answer_type_from_question(text: str, structured_data: list = None, file_links: list = None, media_files: list = None) -> str:
        """Infer expected answer type from question text and context (files, media, structured data)."""
        # Default
        if not text and not (file_links or media_files or structured_data):
            return "string"

        text_lower = (text or "").lower()
        file_links = file_links or []
        media_files = media_files or []

        # Gather extensions present
        def ext_of(url: str) -> str:
            try:
                from urllib.parse import urlparse
                path = urlparse(url).path
                import os as _os
                return _os.path.splitext(path)[1].lower()
            except Exception:
                return ""

        exts = {ext_of(fl) for fl in file_links if fl}
        media_exts = {ext_of(m.get('url') or m.get('src') or '') for m in media_files if m}

        # If any audio present, likely textual transcription
        audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus'}
        if media_exts & audio_exts:
            return "string"

        # If images present with OCR intent, answers are usually textual
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
        if media_exts & image_exts:
            # Prefer textual unless question clearly asks for numeric computation
            if any(k in text_lower for k in ("count", "how many", "number of")):
                return "number"
            return "string"

        # PDF or text documents typically yield a textual answer (password/secret/hidden text)
        doc_exts_string = {'.pdf', '.txt', '.md', '.rtf'}
        if exts & doc_exts_string:
            return "string"

        # Spreadsheets or CSV often numeric, but can be textual; check keywords
        tabular_exts = {'.csv', '.tsv', '.xlsx', '.xls'}
        has_tabular = bool(exts & tabular_exts)
        # JSON may be dict-like answers
        if '.json' in exts:
            # If question asks to "return json/dict", prefer dict
            if any(k in text_lower for k in ('return json', 'return dict', 'as json', 'output json')):
                return "dict"

        # Keywords for audio/transcription -> textual
        audio_keywords = [
            'audio transcription', 'listen to the audio', 'transcribe',
            'secret code in audio', 'speech to text', 'from the audio',
            'what does the audio say', 'audio file', 'voice'
        ]
        if any(keyword in text_lower for keyword in audio_keywords):
            return "string"

        # "code"/"password"/"secret" strongly imply textual
        code_keywords = ['secret code', 'code', 'passcode', 'unlock code', 'access code', 'password', 'key']
        if any(keyword in text_lower for keyword in code_keywords):
            return "string"

        # Numeric indicators (require stronger signals to avoid false positives)
        numeric_keywords = [
            'calculate', 'compute', 'regression',
            'how many', 'count', 'sum', 'average',
            'mean', 'median', 'standard deviation', 'variance', 'correlation',
            'round to', 'decimal place', 'number', 'percentage', 'probability',
            'linear regression', 'train a model', 'value for'
        ]
        strong_numeric = any(k in text_lower for k in numeric_keywords)
        if has_tabular and strong_numeric:
            return "number"

        # Structured data indicating ML training
        if structured_data:
            for block in structured_data:
                if block.get('type') == 'json':
                    data = block.get('data', {})
                    if isinstance(data, dict):
                        if 'X' in data and 'y' in data:
                            return "number"
                        if 'training' in str(data).lower():
                            return "number"

        # Boolean indicators
        boolean_keywords = ['true or false', 'yes or no', 'is it', 'does it', 'will it']
        if any(keyword in text_lower for keyword in boolean_keywords):
            return "boolean"

        # Collection indicators -> dict/structured
        collection_keywords = ['list all', 'find all', 'return json', 'return dict', 'return array']
        if any(keyword in text_lower for keyword in collection_keywords):
            return "dict"

        # Default to string
        return "string"

    # Helper: transcribe audio files using best available backend
    async def transcribe_audio_files(media_files: list, processed_files: list) -> list:
        """
        Transcribe audio media files by mapping to downloaded local paths.
        Cloud backends (preferred, no local deps):
          - OpenAI Whisper API (OPENAI_API_KEY)
          - Deepgram (DEEPGRAM_API_KEY)
          - AssemblyAI (ASSEMBLYAI_API_KEY)
        Local backends (optional if installed):
          - whisper (local)
          - Vosk (requires VOSK_MODEL_PATH)
          - SpeechRecognition + Sphinx
        Returns list of dicts: { url, path, text, backend, error }
        """
        audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus'}
        results = []

        # Collect candidate local files from processed files
        local_files = []
        for pf in processed_files:
            try:
                p = pf.get('path') or pf.get('local_path') or pf.get('file_path')
                if p:
                    local_files.append(Path(p))
            except Exception:
                continue

        # Utilities
        def best_match_by_name(target_basename: str, candidates: list[Path]) -> Optional[str]:
            """Find the best local candidate by filename similarity."""
            if not candidates:
                return None
            try:
                from difflib import SequenceMatcher
                target_stem = Path(target_basename).stem.lower()
                scored = []
                for lf in candidates:
                    if lf.suffix.lower() not in audio_exts:
                        continue
                    score = SequenceMatcher(None, target_stem, lf.stem.lower()).ratio()
                    scored.append((score, lf))
                if not scored:
                    return None
                scored.sort(key=lambda x: x[0], reverse=True)
                # Accept good matches or fallback to the top one
                top_score, top_lf = scored[0]
                return str(top_lf)
            except Exception:
                # Fallback: first audio candidate
                for lf in candidates:
                    if lf.suffix.lower() in audio_exts:
                        return str(lf)
                return None

        def scan_audio_dirs() -> list[str]:
            """Scan default data directories for audio files."""
            found = []
            try:
                base_dirs = ['data/raw', 'data/processed']
                for d in base_dirs:
                    if os.path.isdir(d):
                        for ext in audio_exts:
                            found.extend(Path(d).glob(f'**/*{ext}'))
            except Exception:
                return []
            return [str(p) for p in found]

        def map_media_to_local(m):
            src = m.get('local_path') or m.get('url') or m.get('src') or ''
            if not src:
                return None
            basename = os.path.basename(src.split('?')[0])

            # 1) Try exact/starts-with match among processed audio files
            for lf in local_files:
                try:
                    if lf.suffix.lower() in audio_exts:
                        if os.path.basename(str(lf)).startswith(Path(basename).stem):
                            return str(lf)
                        if os.path.basename(str(lf)) == basename:
                            return str(lf)
                except Exception:
                    continue

            # 2) Try similarity match
            match = best_match_by_name(basename, local_files)
            if match:
                return match

            # 3) Fallback: single audio file in processed list
            audios = [str(lf) for lf in local_files if lf.suffix.lower() in audio_exts]
            if len(audios) == 1:
                return audios[0]

            # 4) Global scan in data directories, prefer newest
            found = scan_audio_dirs()
            if len(found) == 1:
                return found[0]
            elif len(found) > 1:
                found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return found[0]

            return None

        # Detect cloud backends via env
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()

        # Optional imports for local backends (skip if not available)
        whisper = None
        sr = None
        vosk = None
        openai_client = None
        try:
            import whisper as _whisper
            whisper = _whisper
        except Exception:
            whisper = None
        try:
            import speech_recognition as _sr
            sr = _sr
        except Exception:
            sr = None
        try:
            import vosk as _vosk
            vosk = _vosk
        except Exception:
            vosk = None
        try:
            if openai_api_key:
                import openai
                openai_client = openai.OpenAI(api_key=openai_api_key) if hasattr(openai, "OpenAI") else openai
        except Exception:
            openai_client = None

        # Utility: convert audio to mono 16kHz PCM WAV for offline backends (optional; only if libs exist)
        def convert_to_wav_pcm16(src_path: str):
            try:
                try:
                    from pydub import AudioSegment
                    ffmpeg_bin = os.getenv("FFMPEG_BIN")
                    if ffmpeg_bin:
                        AudioSegment.converter = ffmpeg_bin
                        AudioSegment.ffmpeg = ffmpeg_bin
                        AudioSegment.ffprobe = ffmpeg_bin
                    audio = AudioSegment.from_file(src_path)
                    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                    dst_dir = os.path.join("data", "processed")
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, f"{Path(src_path).stem}_mono16k.wav")
                    audio.export(dst_path, format="wav", parameters=["-acodec", "pcm_s16le"])
                    return dst_path if os.path.isfile(dst_path) else None
                except Exception:
                    try:
                        import soundfile as sf
                        import numpy as np
                        data, sr = sf.read(src_path)
                        if getattr(data, "ndim", 1) > 1:
                            data = data.mean(axis=1)
                        target_sr = 16000
                        if sr != target_sr:
                            try:
                                import resampy
                                data = resampy.resample(data, sr, target_sr)
                            except Exception:
                                ratio = target_sr / float(sr)
                                import numpy as np
                                idx = (np.arange(int(len(data) * ratio)) / ratio).astype(np.int64)
                                idx = np.clip(idx, 0, len(data) - 1)
                                data = data[idx]
                        dst_dir = os.path.join("data", "processed")
                        os.makedirs(dst_dir, exist_ok=True)
                        dst_path = os.path.join(dst_dir, f"{Path(src_path).stem}_mono16k.wav")
                        sf.write(dst_path, data, target_sr, subtype="PCM_16")
                        return dst_path if os.path.isfile(dst_path) else None
                    except Exception:
                        return None
            except Exception:
                return None

        # Helper: load audio for Vosk (expects PCM stream)
        def load_audio_for_vosk(path: str):
            try:
                import soundfile as sf
                import numpy as np
                data, samplerate = sf.read(path)
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=1)
                target_sr = 16000
                if samplerate != target_sr:
                    try:
                        import resampy
                        data = resampy.resample(data, samplerate, target_sr)
                    except Exception:
                        ratio = target_sr / float(samplerate)
                        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(np.int64)
                        idx = np.clip(idx, 0, len(data) - 1)
                        data = data[idx]
                # Convert to 16-bit PCM bytes
                data = (data * 32767).astype('int16').tobytes()
                return data, target_sr
            except Exception:
                return None, None

        # Cloud backend helpers (no local deps)
        async def transcribe_with_deepgram(path: str) -> Optional[str]:
            if not deepgram_api_key:
                return None
            try:
                import httpx
                url = "https://api.deepgram.com/v1/listen"
                headers = {"Authorization": f"Token {deepgram_api_key}"}
                params = {
                    "model": os.getenv("DEEPGRAM_MODEL", "nova-2"),
                    "smart_format": "true",
                    "punctuate": "true",
                }
                with open(path, "rb") as f:
                    data = f.read()
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(url, headers=headers, params=params, content=data)
                    resp.raise_for_status()
                    j = resp.json()
                    # Extract transcript generically
                    transcript = (
                        j.get("results", {})
                         .get("channels", [{}])[0]
                         .get("alternatives", [{}])[0]
                         .get("transcript")
                    )
                    return (transcript or "").strip() or None
            except Exception:
                return None

        async def transcribe_with_assemblyai(path: str) -> Optional[str]:
            if not assemblyai_api_key:
                return None
            try:
                import httpx
                headers = {"authorization": assemblyai_api_key}
                # Upload
                async with httpx.AsyncClient(timeout=60) as client:
                    with open(path, "rb") as f:
                        upload_resp = await client.post("https://api.assemblyai.com/v2/upload", headers=headers, content=f)
                    upload_resp.raise_for_status()
                    upload_url = upload_resp.json().get("upload_url")
                    if not upload_url:
                        return None
                    # Request transcription
                    body = {
                        "audio_url": upload_url,
                        "punctuate": True,
                        "format_text": True
                    }
                    tr_resp = await client.post("https://api.assemblyai.com/v2/transcribe", headers=headers, json=body)
                    tr_resp.raise_for_status()
                    tr_id = tr_resp.json().get("id")
                    if not tr_id:
                        return None
                    # Poll status
                    poll_url = f"https://api.assemblyai.com/v2/transcribe/{tr_id}"
                    for _ in range(int(os.getenv("ASSEMBLYAI_MAX_POLLS", "30"))):
                        s = await client.get(poll_url, headers=headers)
                        s.raise_for_status()
                        js = s.json()
                        if js.get("status") == "completed":
                            return (js.get("text") or "").strip() or None
                        if js.get("status") in ("error", "failed"):
                            break
                        await asyncio.sleep(1.5)
                return None
            except Exception:
                return None

        async def transcribe_with_openai(path: str) -> Optional[str]:
            try:
                if openai_client:
                    if hasattr(openai_client, "audio") and hasattr(openai_client.audio, "transcriptions"):
                        with open(path, "rb") as f:
                            resp = openai_client.audio.transcriptions.create(
                                model=os.getenv('OPENAI_WHISPER_API_MODEL', 'whisper-1'),
                                file=f
                            )
                        t = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
                        return (t or "").strip() or None
                    else:
                        with open(path, "rb") as f:
                            t = openai_client.Audio.transcribe(os.getenv('OPENAI_WHISPER_API_MODEL', 'whisper-1'), f).get("text")
                        return (t or "").strip() or None
                return None
            except Exception:
                return None

        import asyncio

        for m in media_files or []:
            try:
                url = m.get('url') or m.get('src')
                if not url:
                    continue
                if Path(url.split('?')[0]).suffix.lower() not in audio_exts:
                    continue

                local_path = m.get('local_path') or map_media_to_local(m)
                if not local_path or not os.path.isfile(local_path):
                    candidates = scan_audio_dirs()
                    if candidates:
                        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
                        local_path = candidates[0] if candidates else None

                if not local_path or not os.path.isfile(local_path):
                    results.append({'url': url, 'path': local_path, 'text': None, 'backend': None, 'error': 'local audio not found'})
                    continue

                text = None
                backend_used = None
                errors = []

                # Prefer cloud backends to avoid local dependencies
                if not text and openai_api_key:
                    t = await transcribe_with_openai(local_path)
                    if t:
                        text, backend_used = t, 'openai:whisper'
                    else:
                        errors.append('openai whisper failed or returned empty')

                if not text and deepgram_api_key:
                    t = await transcribe_with_deepgram(local_path)
                    if t:
                        text, backend_used = t, 'deepgram'
                    else:
                        errors.append('deepgram failed or returned empty')

                if not text and assemblyai_api_key:
                    t = await transcribe_with_assemblyai(local_path)
                    if t:
                        text, backend_used = t, 'assemblyai'
                    else:
                        errors.append('assemblyai failed or returned empty')

                # Local backends (optional)
                if whisper and not text:
                    try:
                        model_name = os.getenv('WHISPER_MODEL', 'base')
                        model = whisper.load_model(model_name)
                        trans = model.transcribe(local_path, fp16=False)
                        t = (trans.get('text') or '').strip()
                        if t:
                            text, backend_used = t, f'whisper:{model_name}'
                        else:
                            errors.append('whisper produced empty text')
                    except Exception as e:
                        errors.append(f'whisper failed: {e}')

                wav_path = convert_to_wav_pcm16(local_path) if (vosk or sr) and not text else None

                if vosk and not text:
                    try:
                        model_path = os.getenv('VOSK_MODEL_PATH', '').strip()
                        if model_path and os.path.isdir(model_path):
                            from vosk import Model, KaldiRecognizer
                            pcm_bytes, sr_hz = load_audio_for_vosk(wav_path or local_path)
                            if pcm_bytes and sr_hz:
                                model = Model(model_path)
                                rec = KaldiRecognizer(model, sr_hz)
                                rec.SetWords(True)
                                chunk_size = 4000
                                for i in range(0, len(pcm_bytes), chunk_size):
                                    rec.AcceptWaveform(pcm_bytes[i:i+chunk_size])
                                final = rec.FinalResult()
                                import json as _json
                                try:
                                    parsed = _json.loads(final or '{}')
                                    t = (parsed.get('text') or '').strip()
                                except Exception:
                                    t = (final or '').strip()
                                if t:
                                    text, backend_used = t, 'vosk'
                                else:
                                    errors.append('vosk produced empty text')
                        else:
                            errors.append('vosk model path missing or invalid (set VOSK_MODEL_PATH)')
                    except Exception as e:
                        errors.append(f'vosk failed: {e}')

                if sr and not text:
                    try:
                        recognizer = sr.Recognizer()
                        use_path = wav_path or local_path
                        with sr.AudioFile(use_path) as source:
                            audio_data = recognizer.record(source)
                        try:
                            t = recognizer.recognize_sphinx(audio_data)
                            t = (t or '').strip()
                            if t:
                                text, backend_used = t, 'speech_recognition:sphinx'
                            else:
                                errors.append('sphinx produced empty text')
                        except Exception as e:
                            errors.append(f'sphinx unavailable: {e}')
                    except Exception as e:
                        errors.append(f'speech_recognition failed: {e}')

                error = None
                if not text:
                    error = '; '.join(errors) if errors else 'no transcription backend succeeded'

                results.append({'url': url, 'path': local_path, 'text': text or None, 'backend': backend_used, 'error': error})
            except Exception as e:
                results.append({'url': m.get('url'), 'path': None, 'text': None, 'backend': None, 'error': str(e)})

        return results

    # Main quiz loop - handles multiple levels
    while current_url:
        # Restart timer for each level
        level_start_time = time.time()
        level_time_limit = 170.0  # 170 seconds per level (leaving 10s buffer)
        
        logger.info(f"{'='*60}")
        logger.info(f"LEVEL {level} - Processing: {current_url}")
        logger.info(f"Time limit for this level: {level_time_limit}s")
        logger.info(f"{'='*60}")
        
        expected_answer_type = "string"  # Default
        quiz_result = None  # Store fetch result for context
        
        try:
            # Compute remaining time for this level (used by asyncio.wait_for across steps)
            def time_remaining() -> float:
                rem = level_time_limit - (time.time() - level_start_time)
                # Keep a small floor to avoid zero/negative timeouts
                return max(0.1, rem)

            # Step 1: Fetch and render the quiz page
            logger.info("Step 1: Fetching quiz page...")
            try:
                result = await asyncio.wait_for(fetch_and_render_quiz_page(current_url), timeout=time_remaining())
            except asyncio.TimeoutError:
                logger.error("Step 1 timed out; submitting dummy to advance")
                # Submit dummy answer and try to continue
                dummy_result = await submit_dummy_answer(
                    submit_url=None,
                    email=email,
                    secret=secret,
                    current_url=current_url,
                    expected_type="string"
                )
                if dummy_result.get('next_url'):
                    current_url = dummy_result['next_url']; level += 1
                    continue
                break
            quiz_result = result  # Store for later use
            
            if not result['success']:
                logger.error(f"Failed to fetch quiz page: {result['error']}")
                # Submit dummy answer and try to continue
                if result.get('submit_url'):
                    dummy_result = await submit_dummy_answer(
                        submit_url=result['submit_url'],
                        email=email,
                        secret=secret,
                        current_url=current_url,
                        expected_type="string"
                    )
                    if dummy_result.get('next_url'):
                        current_url = dummy_result['next_url']
                        level += 1
                        continue
                break
            
            logger.info(f"Question: {result['question']}")
            logger.info(f"Found {len(result['download_links'])} files, {len(result.get('media_files', []))} media")
            
            # Infer expected answer type from question text and structured data
            expected_answer_type = infer_answer_type_from_question(
                result.get('text', ''),
                result.get('structured_data', []),
                file_links=[link['url'] for link in result.get('download_links', [])],
                media_files=result.get('media_files', [])
            )
            logger.info(f"Inferred expected answer type: {expected_answer_type}")
            
            # Log structured data if present
            if result.get('structured_data'):
                logger.info(f"Found {len(result['structured_data'])} structured data blocks")
                for idx, block in enumerate(result['structured_data'], 1):
                    block_id = f" (id={block['id']})" if block.get('id') else ""
                    logger.info(f"  Block {idx}: {block['type']}{block_id}")
            
            # Debug logging for file types
            if result['download_links']:
                logger.info(f"Download links: {[link['url'] for link in result['download_links']]}")
            if result.get('media_files'):
                logger.info(f"Media files: {[m['url'] for m in result['media_files']]}")
            
            submit_url = result['submit_url']
            if not submit_url:
                logger.error("No submit URL found")
                break
            
            # Retry loop for current level
            max_retries_per_level = 2
            retry_count = 0
            failure_reason = None
            
            while retry_count <= max_retries_per_level:
                elapsed = time.time() - level_start_time
                
                # Check time limit for this level
                if elapsed >= level_time_limit:
                    logger.error(f"Time limit exceeded for level {level} ({elapsed:.1f}s >= {level_time_limit}s)")
                    break
                
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries_per_level}")
                    logger.info(f"Time elapsed this level: {elapsed:.1f}s / {level_time_limit}s")
                
                try:
                    # Step 2: Download and process data files
                    logger.info("Step 2: Processing data files...")
                    
                    # Combine download links and media files for processing
                    all_file_urls = [link['url'] for link in result['download_links']]
                    media_urls = [m['url'] for m in result.get('media_files', [])]
                    
                    # Add media files to processing list
                    all_file_urls.extend(media_urls)
                    
                    process_result = await asyncio.wait_for(
                        process_all_files(
                            file_links=all_file_urls,
                            preprocessing_instructions=result.get('text', ''),
                            failure_reason=failure_reason
                        ),
                        timeout=time_remaining()
                    )
                    
                    if not process_result['success']:
                        logger.error(f"Data processing failed: {process_result['errors']}")
                        raise Exception(f"Data processing failed: {', '.join(process_result['errors'][:3])}")
                    
                    logger.info(f"Processed {len(process_result['processed_files'])} files")
                    
                    # Step 2.5: Extract text from images
                    logger.info("Step 2.5: Extracting text from images...")
                    image_text_extractions = []
                    
                    if result.get('media_files'):
                        try:
                            image_text_extractions = await asyncio.wait_for(
                                extract_text_from_images(
                                    media_files=result['media_files'],
                                    processed_files=process_result['processed_files']
                                ),
                                timeout=time_remaining()
                            )
                        except asyncio.TimeoutError:
                            logger.error("Image extraction timed out; proceeding without OCR")
                            image_text_extractions = []
                        # Propagate local paths from extraction into media_files for later stages
                        if image_text_extractions:
                            for m in result['media_files']:
                                basename = os.path.basename(m.get('url','').split('?')[0])
                                match = next((ex for ex in image_text_extractions if ex.get('path') and os.path.basename(ex['path']).startswith(Path(basename).stem)), None)
                                if match and match.get('path'):
                                    m['local_path'] = match['path']
                        
                        if image_text_extractions:
                            logger.info(f"Extracted text from {len(image_text_extractions)} images")
                            for idx, extraction in enumerate(image_text_extractions):
                                if extraction.get('text'):
                                    logger.info(f"Image {idx+1} text: {extraction['text'][:100]}...")
                                else:
                                    logger.warning(f"Image {idx+1} has no text")
                        else:
                            logger.warning("No text extracted from images")
                    
                    # Step 2.6: Transcribe audio media files
                    logger.info("Step 2.6: Transcribing audio files...")
                    audio_transcriptions = []
                    try:
                        # Ensure media entries have local_path propagated similar to images
                        if result.get('media_files'):
                            # Attempt to map local paths based on processed_files
                            for m in result['media_files']:
                                if not m.get('local_path'):
                                    basename = os.path.basename((m.get('url') or '').split('?')[0])
                                    match = next(
                                        (pf for pf in process_result['processed_files']
                                         if pf.get('path') and (
                                             os.path.basename(pf['path']).startswith(Path(basename).stem) or
                                             os.path.basename(pf['path']) == basename
                                         )),
                                        None
                                    )
                                    if match and match.get('path'):
                                        m['local_path'] = match['path']

                            audio_transcriptions = await asyncio.wait_for(
                                transcribe_audio_files(
                                    media_files=result['media_files'],
                                    processed_files=process_result['processed_files']
                                ),
                                timeout=time_remaining()
                            )

                            # Log brief transcription info and inject into structured_data
                            if audio_transcriptions:
                                success_count = sum(1 for t in audio_transcriptions if t.get('text'))
                                logger.info(f"Transcribed {success_count}/{len(audio_transcriptions)} audio files")
                                for idx, tr in enumerate(audio_transcriptions):
                                    preview = (tr.get('text') or '')[:100]
                                    if tr.get('text'):
                                        logger.info(f"Audio {idx+1} ({tr.get('backend')}): {preview}...")
                                    elif tr.get('error'):
                                        logger.warning(f"Audio {idx+1} transcription error: {tr['error']}")
                                # Extend structured_data with transcription blocks for downstream LLM
                                if not result.get('structured_data'):
                                    result['structured_data'] = []
                                for tr in audio_transcriptions:
                                    result['structured_data'].append({
                                        'type': 'audio_transcription',
                                        'id': f"audio:{os.path.basename((tr.get('path') or tr.get('url') or ''))}",
                                        'data': {
                                            'url': tr.get('url'),
                                            'path': tr.get('path'),
                                            'text': tr.get('text'),
                                            'backend': tr.get('backend'),
                                            'error': tr.get('error')
                                        }
                                    })
                                # If any transcription exists, prefer string answers
                                if any(t.get('text') for t in audio_transcriptions):
                                    expected_answer_type = 'string'
                            else:
                                logger.warning("No audio files to transcribe or mapping failed")
                        else:
                            logger.info("No media files present for audio transcription")
                    except Exception as e:
                        logger.error(f"Audio transcription step failed: {e}")
                        audio_transcriptions = []

                    # Step 3: Generate answer using LLM
                    logger.info("Step 3: Generating answer...")
                    try:
                        answer_result = await asyncio.wait_for(
                            generate_quiz_answer(
                                instructions=result.get('text', ''),
                                processed_data=process_result['processed_files'],
                                page_html=result.get('html', ''),
                                media_files=result.get('media_files', []),
                                image_text_extractions=image_text_extractions,
                                failure_reason=failure_reason,
                                structured_data=result.get('structured_data', [])
                            ),
                            timeout=time_remaining()
                        )
                    except asyncio.TimeoutError:
                        logger.error("Answer generation timed out; submitting dummy immediately")
                        # Re-infer answer type from question if we haven't gotten it from LLM
                        expected_answer_type = infer_answer_type_from_question(
                            quiz_result.get('text', ''),
                            quiz_result.get('structured_data', [])
                        ) if quiz_result else expected_answer_type
                        dummy_result = await submit_dummy_answer(
                            submit_url=submit_url,
                            email=email,
                            secret=secret,
                            current_url=current_url,
                            expected_type=expected_answer_type
                        )
                        if dummy_result.get('next_url'):
                            current_url = dummy_result['next_url']; level += 1
                            break
                        else:
                            current_url = None
                            break

                    # Validate generation result before using
                    if not answer_result.get('success') or ('answer' not in answer_result):
                        logger.error(f"Answer generation failed: {answer_result.get('error')}")
                        # Re-infer expected type for dummy submission using full context
                        expected_answer_type = infer_answer_type_from_question(
                            quiz_result.get('text', ''),
                            quiz_result.get('structured_data', []),
                            file_links=[link['url'] for link in quiz_result.get('download_links', [])] if quiz_result else [],
                            media_files=quiz_result.get('media_files', []) if quiz_result else []
                        ) if quiz_result else expected_answer_type
                        logger.warning("Submitting dummy answer due to failed generation...")
                        dummy_result = await submit_dummy_answer(
                            submit_url=submit_url,
                            email=email,
                            secret=secret,
                            current_url=current_url,
                            expected_type=expected_answer_type
                        )
                        if dummy_result.get('next_url'):
                            current_url = dummy_result['next_url']; level += 1
                            break
                        else:
                            current_url = None
                            break

                    logger.info(f"Answer generated: {answer_result.get('answer_type', 'auto')}")
                    # Update expected type from actual answer if available
                    if answer_result.get('answer_type'):
                        expected_answer_type = answer_result['answer_type']
                    
                    # Step 4: Submit answer
                    logger.info("Step 4: Submitting answer...")
                    try:
                        submission_result = await asyncio.wait_for(
                            submit_answer(
                                submit_url=submit_url,
                                email=email,
                                secret=secret,
                                current_url=current_url,
                                answer=answer_result['answer']
                            ),
                            timeout=time_remaining()
                        )
                    except asyncio.TimeoutError:
                        logger.error("Submission timed out; submitting dummy to advance")
                        dummy_result = await submit_dummy_answer(
                            submit_url=submit_url,
                            email=email,
                            secret=secret,
                            current_url=current_url,
                            expected_type=expected_answer_type
                        )
                        if dummy_result.get('next_url'):
                            current_url = dummy_result['next_url']; level += 1
                            break
                        else:
                            current_url = None
                            break

                    # If submission failed with 404, try alternate URL
                    if not submission_result['success'] and '404' in str(submission_result.get('error', '')):
                        logger.warning("Got 404, checking for alternate submit URLs...")
                        
                        # Try to extract alternate URL from current page URL
                        from urllib.parse import urlparse, urljoin
                        parsed = urlparse(current_url)
                        base_url = f"{parsed.scheme}://{parsed.netloc}"
                        
                        # Try common API endpoint patterns
                        alternate_urls = [
                            urljoin(base_url, '/api/submit'),
                            urljoin(base_url, '/api/answer'),
                            urljoin(base_url, '/submit'),
                            urljoin(base_url, '/answer'),
                            urljoin(base_url, '/api/quiz/submit'),
                        ]
                        
                        for alt_url in alternate_urls:
                            if alt_url == submit_url:
                                continue
                            
                            logger.info(f"Trying alternate URL: {alt_url}")
                            try:
                                submission_result = await asyncio.wait_for(
                                    submit_answer(
                                        submit_url=alt_url,
                                        email=email,
                                        secret=secret,
                                        current_url=current_url,
                                        answer=answer_result['answer']
                                    ),
                                    timeout=time_remaining()
                                )
                            except asyncio.TimeoutError:
                                logger.error("Alternate submission timed out; using dummy")
                                dummy_result = await submit_dummy_answer(
                                    submit_url=submit_url,
                                    email=email,
                                    secret=secret,
                                    current_url=current_url,
                                    expected_type=expected_answer_type
                                )
                                if dummy_result.get('next_url'):
                                    current_url = dummy_result['next_url']; level += 1
                                    break
                                else:
                                    current_url = None
                                    break
                            # ...existing code handling submission_result...
                            if submission_result['success']:
                                logger.info(f"Success with alternate URL: {alt_url}")
                                break
                    
                    if not submission_result['success']:
                        logger.error(f"Submission failed: {submission_result['error']}")
                        raise Exception(f"Submission failed: {submission_result['error']}")
                    
                    # Step 5: Handle submission response
                    if submission_result['correct']:
                        level_elapsed = time.time() - level_start_time
                        logger.info(f"✓ CORRECT! Level {level} completed in {level_elapsed:.1f}s")
                        
                        # Check for next level
                        next_url = submission_result.get('next_url')
                        if next_url:
                            logger.info(f"Moving to next level: {next_url}")
                            current_url = next_url
                            level += 1
                            break  # Break retry loop, continue main loop
                        else:
                            logger.info("🎉 QUIZ COMPLETED!")
                            current_url = None
                            break
                    
                    else:
                        # Answer was incorrect
                        failure_reason = submission_result.get('reason', 'Answer was incorrect')
                        logger.warning(f"✗ INCORRECT: {failure_reason}")
                        
                        # Decide whether to retry
                        elapsed = time.time() - level_start_time
                        if should_retry(submission_result, elapsed, level_time_limit):
                            retry_count += 1
                            logger.info(f"Will retry (attempt {retry_count + 1})")
                            continue  # Continue retry loop
                        else:
                            logger.info("Insufficient time for retry")
                            next_url = submission_result.get('next_url')
                            if next_url:
                                logger.info(f"Moving to next level: {next_url}")
                                current_url = next_url
                                level += 1
                                break
                            else:
                                current_url = None
                                break
                
                except Exception as step_error:
                    logger.error(f"Error in quiz steps: {step_error}")
                    
                    # Re-infer answer type from question if we haven't gotten it from LLM
                    if quiz_result:
                        expected_answer_type = infer_answer_type_from_question(
                            quiz_result.get('text', ''),
                            quiz_result.get('structured_data', []),
                            file_links=[link['url'] for link in quiz_result.get('download_links', [])] if quiz_result else [],
                            media_files=quiz_result.get('media_files', []) if quiz_result else []
                        ) if quiz_result else expected_answer_type
                    
                    # Submit dummy answer as fallback
                    logger.warning("Submitting dummy answer...")
                    # Submit dummy answer as fallback immediately (respecting time limit)
                    try:
                        dummy_result = await asyncio.wait_for(
                            submit_dummy_answer(
                                submit_url=submit_url,
                                email=email,
                                secret=secret,
                                current_url=current_url,
                                expected_type=expected_answer_type
                            ),
                            timeout=time_remaining()
                        )
                        
                        if dummy_result.get('success'):
                            next_url = dummy_result.get('next_url')
                            if next_url:
                                logger.info(f"Moving to: {next_url}")
                                current_url = next_url
                                level += 1
                                break
                            else:
                                current_url = None
                                break
                        else:
                            logger.error("Dummy submission also failed")
                            current_url = None
                            break
                    
                    except asyncio.TimeoutError:
                        logger.error("Dummy submission timed out; stopping level")
                        current_url = None
                        break
            
            # If we exhausted retries without success
            if retry_count > max_retries_per_level and current_url:
                logger.warning(f"Exhausted retries for level {level}")
                # Re-infer answer type
                if quiz_result:
                    expected_answer_type = infer_answer_type_from_question(
                        quiz_result.get('text', ''),
                        quiz_result.get('structured_data', []),
                        file_links=[link['url'] for link in quiz_result.get('download_links', [])] if quiz_result else [],
                        media_files=quiz_result.get('media_files', []) if quiz_result else []
                    )
                
                # Submit dummy answer to move forward
                logger.info("Submitting dummy answer to continue...")
                try:
                    dummy_result = await asyncio.wait_for(
                        submit_dummy_answer(
                            submit_url=submit_url,
                            email=email,
                            secret=secret,
                            current_url=current_url,
                            expected_type=expected_answer_type
                        ),
                        timeout=time_remaining()
                    )
                    
                    if dummy_result.get('success'):
                        next_url = dummy_result.get('next_url')
                        if next_url:
                            logger.info(f"Moving to next level: {next_url}")
                            current_url = next_url
                            level += 1
                            continue  # Continue to next level
                        else:
                            logger.info("No next level available")
                            current_url = None
                    else:
                        logger.error("Dummy submission failed, stopping")
                        current_url = None
                except asyncio.TimeoutError:
                    logger.error("Final dummy submission timed out; stopping")
                    current_url = None
        
        except Exception as level_error:
            logger.error(f"Critical error at level {level}: {level_error}")
            
            # Try to infer answer type from any available context
            if quiz_result:
                expected_answer_type = infer_answer_type_from_question(
                    quiz_result.get('text', ''),
                    quiz_result.get('structured_data', [])
                )
            
            # Try dummy submission
            try:
                if 'submit_url' in locals():
                    dummy_result = await asyncio.wait_for(
                        submit_dummy_answer(
                            submit_url=submit_url,
                            email=email,
                            secret=secret,
                            current_url=current_url,
                            expected_type=expected_answer_type
                        ),
                        timeout=time_remaining()
                    )
                    
                    if dummy_result.get('next_url'):
                        current_url = dummy_result['next_url']
                        level += 1
                        continue
            except:
                pass
            break
    
    logger.info(f"{'='*60}")
    logger.info(f"Quiz completed - {level} levels attempted")
    logger.info(f"{'='*60}")


# POST endpoint for quiz solving
@app.post("/solve-quiz", status_code=status.HTTP_200_OK)
async def solve_quiz(request: QuizSolveRequest, background_tasks: BackgroundTasks):
    # Validate secret
    if request.secret != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Invalid secret"}
        )
    
    # Add background task with secret parameter
    background_tasks.add_task(run_quiz_process, request.url, request.email, request.secret)
    
    # Return immediate response
    return {"message": "Quiz processing started"}


if __name__ == "__main__":
    import uvicorn
    # Run with: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
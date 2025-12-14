# OpenAI API Compatible SenseVoice STT API

Offline speech recognition service based on SenseVoice, providing OpenAI Whisper API compatible HTTP endpoints.
Modified based on [FunASR HTTP Server](https://github.com/modelscope/FunASR/tree/main/runtime/python/http)

## Features

- ✅ **OpenAI API Compatible**: Supports `/v1/audio/transcriptions` endpoint
- ✅ **Multiple Response Formats**: JSON, Text, Verbose JSON, SRT, VTT
- ✅ **Multi-language Support**: Chinese, English, Cantonese, Japanese, Korean
- ✅ **Auto Language Detection**: Supports `language=auto`
- ✅ **VAD Segmentation**: Automatic voice activity detection and segment merging
- ✅ **Inverse Text Normalization**: Supports number, date formatting in output
- ✅ **High Performance**: Uses `soundfile` + `torchaudio` for in-memory audio processing, eliminating disk I/O overhead
- ✅ **LLM Post-Processing**: Optional 2-pass mode with LLM-based transcript correction for improved accuracy 

## Installation

```bash
uv sync
```

### CUDA Support

If you want to accelerate the inference using your GPU with CUDA support, modify the `pyproject.toml` by adding the following lines to it.
Adjust the CUDA version according to your GPU. You may also need to adjust the `torch`, `torchvision`, `torchaudio` versions.

```
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }
torchaudio = { index = "pytorch" }
```

## Deployment

### Basic Startup

```bash
uv run funasr_http_server.py --port 8200 --device cpu
```

### Run in background

```bash
nohup uv run funasr_http_server.py --port 8000 --device mps > server.log 2>&1 &
```

### Custom Configuration

```bash
uv run funasr_http_server.py \
  --port 8200 \
  --device cpu \
  --language yue \
  --model_dir ./models/iic/SenseVoiceSmall \
  --use_itn True \
  --merge_vad True \
  --merge_length_s 15
```

### LLM Post-Processing (2-Pass Mode)

Enable optional LLM-based transcript correction for improved accuracy:

```bash
uv run funasr_http_server.py --port 8000 --device cpu --llm_correct
```

**Configuration:**

1. Create a `.env` file in the project root:
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

2. (Optional) Customize the system prompt by creating `prompts/llm_correction_system.txt`:
```
You are an expert editor designed to post-process ASR transcripts.
Your task is to correct spelling, grammar, and punctuation...
```

**Notes:**
- If `prompts/llm_correction_system.txt` exists, it will be used as the system prompt
- Otherwise, a sensible default prompt is used
- Works with OpenAI-compatible APIs (OpenAI, Azure, local LLMs, etc.)
- The LLM correction applies to the `/v1/audio/transcriptions` endpoint

### Docker Deployment

#### Build Docker Image

```bash
docker build -t sensevoice-stt:latest .
```

#### Run with Model Auto-Download

Run the container with a volume mount for model persistence. The model will be automatically downloaded from ModelScope to the mounted volume on first run:

```bash
# Create models directory (if not exists)
mkdir -p ./models

# Run container with model volume mount
docker run -d \
  --name sensevoice-stt \
  -p 8200:8200 \
  -v $PWD/models:/root/.cache/modelscope/hub \
  sensevoice-stt:latest
```

**Note:** 
- The model will download to `./models/iic/SenseVoiceSmall/` on first run
- If you already have the model, place it at `./models/iic/SenseVoiceSmall/` before running

**Directory Structure:**
```
./models/
└── iic/
    └── SenseVoiceSmall/
        ├── configuration.json
        ├── model.pt
        └── ...
```

#### Run with Local Model

If you already have the model downloaded locally:

```bash
docker run -d \
  --name sensevoice-stt \
  -p 8200:8200 \
  -v /path/to/your/models:/root/.cache/modelscope/hub \
  sensevoice-stt:latest
```

**Note:** Your local model should be at `/path/to/your/models/iic/SenseVoiceSmall/`.

#### Run with Custom Parameters

```bash
docker run -d \
  --name sensevoice-stt \
  -p 8200:8200 \
  -v $PWD/models:/root/.cache/modelscope/hub \
  sensevoice-stt:latest \
  uv run funasr_http_server.py --language zh --device cpu
```

#### View Container Logs

```bash
docker logs -f sensevoice-stt
```

#### Stop and Remove Container

```bash
docker stop sensevoice-stt
docker rm sensevoice-stt
```

#### Using Docker Compose

For easier deployment, you can use Docker Compose:

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

The `docker-compose.yml` configuration:
- Automatically builds the image
- Mounts `./models` to ModelScope's cache directory
- Models download to `./models/iic/SenseVoiceSmall/`
- Exposes port 8200
- Restarts automatically unless stopped

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--host` | str | `0.0.0.0` | Server listening address |
| `--port` | int | `8200` | Server port |
| `--model_dir` | str | `iic/SenseVoiceSmall` | SenseVoice model directory |
| `--remote_code` | str | `./model.py` | Remote code path |
| `--vad_model` | str | `fsmn-vad` | VAD model name |
| `--vad_kwargs` | int | `30000` | VAD max single segment time (milliseconds) |
| `--device` | str | `cpu` | Device to run on (`cpu` or `cuda` or `mps`) |
| `--ncpu` | int | `4` | Number of CPU cores |
| `--language` | str | `auto` | Language code (`auto`, `zh`, `en`, `yue`, `ja`, `ko`) |
| `--use_itn` | bool | `True` | Use inverse text normalization |
| `--merge_vad` | bool | `True` | Merge VAD segments |
| `--merge_length_s` | int | `15` | VAD merge max length (seconds) |
| `--llm_correct` | flag | `False` | Enable LLM-based transcript post-processing (requires `.env` config) |

## Usage

### 1. OpenAI API Compatible Endpoint

#### Basic Usage (JSON Format)

```bash
curl --request POST \
  --url http://localhost:8200/v1/audio/transcriptions \
  --header 'Content-Type: multipart/form-data' \
  --form file=@audio/example.wav \
  --form model=sensevoice
```

**Response:**
```json
{"text":"欢迎大家来体验达摩院推出的语音识别模型。"}
```

#### Plain Text Format

```bash
curl --request POST \
  --url http://localhost:8200/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=sensevoice \
  --form response_format=text
```

**Response:**
```
Welcome everyone to experience the speech recognition model launched by DAMO Academy.
```

#### Verbose JSON Format

```bash
curl --request POST \
  --url http://localhost:8200/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=sensevoice \
  --form response_format=verbose_json \
  --form language=zh
```

**Response:**
```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": null,
  "text": "欢迎大家来体验达摩院推出的语音识别模型。"
}
```

#### SRT Subtitle Format

```bash
curl --request POST \
  --url http://localhost:8200/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=sensevoice \
  --form response_format=srt
```

**Response:**
```
1
00:00:00,000 --> 00:00:10,000
Welcome everyone to experience the speech recognition model launched by DAMO Academy.
```

#### VTT Subtitle Format

```bash
curl --request POST \
  --url http://localhost:8200/v1/audio/transcriptions \
  --form file=@audio/example.wav \
  --form model=sensevoice \
  --form response_format=vtt
```

**Response:**
```
WEBVTT

1
00:00:00.000 --> 00:00:10.000
Welcome everyone to experience the speech recognition model launched by DAMO Academy.
```

### 2. Legacy Endpoint (Backward Compatible)

```bash
curl --request POST \
  --url http://localhost:8200/recognition \
  --header 'Content-Type: multipart/form-data' \
  --form audio=@audio/example.wav
```

**Response:**
```json
{
  "text": "欢迎大家来体验达摩院推出的语音识别模型。",
  "code": 0
}
```

### 3. Python Client Example

```python
import requests

url = "http://localhost:8200/v1/audio/transcriptions"

# JSON format
with open("audio/example.wav", "rb") as audio_file:
    files = {"file": audio_file}
    data = {
        "model": "sensevoice",
        "language": "zh",
        "response_format": "json"
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()
    print(result["text"])

# Text format
with open("audio/example.wav", "rb") as audio_file:
    files = {"file": audio_file}
    data = {
        "model": "sensevoice",
        "response_format": "text"
    }
    response = requests.post(url, files=files, data=data)
    print(response.text)
```

## API Reference

### POST /v1/audio/transcriptions

OpenAI Whisper API compatible transcription endpoint.

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ | - | Audio file (supports mp3, wav, m4a, webm, etc.) |
| `model` | string | ✅ | - | Model name (any string) |
| `language` | string | ❌ | `auto` | Language code (ISO-639-1) |
| `prompt` | string | ❌ | - | Optional prompt (currently unused) |
| `response_format` | string | ❌ | `json` | Response format (`json`, `text`, `verbose_json`, `srt`, `vtt`) |
| `temperature` | float | ❌ | `0` | Sampling temperature (currently unused) |

**Response Formats:**

Returns different formats based on `response_format` parameter:
- `json`: `{"text": "..."}`
- `text`: Plain text
- `verbose_json`: `{"task": "...", "language": "...", "duration": ..., "text": "..."}`
- `srt`: SubRip subtitle format
- `vtt`: WebVTT subtitle format

### POST /recognition

Legacy endpoint, maintains backward compatibility.

**Request Parameters:**
- `audio`: Audio file

**Response Format:**
```json
{
  "text": "Recognition result",
  "code": 0
}
```

## Supported Languages

| Language | Code | Status |
|----------|------|--------|
| Auto Detect | `auto` | ✅ |
| Chinese | `zh` | ✅ |
| English | `en` | ✅ |
| Cantonese | `yue` | ✅ |
| Japanese | `ja` | ✅ |
| Korean | `ko` | ✅ |

## Performance Benchmarks

You can benchmark the server using the included `test_speed.py`:

```bash
# Start the server
uv run funasr_http_server.py --port 8000 --device cuda  # or --device mps for Mac

# In another terminal, run the speed test
python test_speed.py audio/your_audio.wav --runs 10
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using the port
lsof -ti:8200

# Kill the process
lsof -ti:8200 | xargs kill -9
```

### Model Loading Failed

Verify the model directory structure is correct:
```
models/iic/SenseVoiceSmall/
├── configuration.json
├── model.pt
└── ...
```

### Audio Processing Falls Back to ffmpeg

If you see warnings like `"torchaudio 處理失敗...回退到 ffmpeg"` in the logs:

1. **Normal behavior**: The server tries `soundfile` first, then falls back to `ffmpeg` for unsupported formats
2. **Performance impact**: Fallback is slightly slower but still works correctly
3. **Solution**: Ensure `soundfile` is installed via `uv sync`

Most common audio formats (WAV, MP3, FLAC, OGG) are supported by `soundfile` and won't trigger fallback.

### ffmpeg Error

The server keeps `ffmpeg` as a fallback for rare audio formats. Ensure ffmpeg is installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

## License

This project is open source. Please refer to the LICENSE file for details.

## Credits

- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- OpenAI Whisper API

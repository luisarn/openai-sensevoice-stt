import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import ffmpeg
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from modelscope.utils.logger import get_logger

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = get_logger(log_level=logging.INFO)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=8000, required=False, help="server port")
parser.add_argument(
    "--model_dir",
    type=str,
    default="models/iic/SenseVoiceSmall",
    help="SenseVoice model directory path",
)
parser.add_argument(
    "--remote_code",
    type=str,
    default="./model.py",
    help="Path to remote code file for SenseVoice model",
)
parser.add_argument(
    "--vad_model",
    type=str,
    default="fsmn-vad",
    help="VAD model name",
)
parser.add_argument(
    "--vad_kwargs",
    type=int,
    default=30000,
    help="max_single_segment_time for VAD in milliseconds",
)
parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--language",
    type=str,
    default="auto",
    help="Language for ASR (auto, zh, en, yue, ja, ko)",
)
parser.add_argument(
    "--use_itn",
    type=bool,
    default=True,
    help="Use inverse text normalization",
)
parser.add_argument(
    "--merge_vad",
    type=bool,
    default=True,
    help="Merge VAD segments",
)
parser.add_argument(
    "--merge_length_s",
    type=int,
    default=15,
    help="Maximum length in seconds for merging VAD segments",
)
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")
parser.add_argument("--temp_dir", type=str, default="temp_dir/", required=False, help="temp dir")
args = parser.parse_args()
logger.info("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    logger.info("%s: %s" % (arg, value))
logger.info("------------------------------------------------")

os.makedirs(args.temp_dir, exist_ok=True)

logger.info("model loading")
# load SenseVoice model
model_dir = Path.cwd() / args.model_dir
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code=args.remote_code,
    vad_model=args.vad_model,
    vad_kwargs={"max_single_segment_time": args.vad_kwargs},
    device=args.device,
    ncpu=args.ncpu,
    disable_pbar=True,
    disable_log=True,
)
logger.info("loaded models!")

app = FastAPI(title="FunASR")

param_dict = {
    "language": args.language,
    "use_itn": args.use_itn,
    "merge_vad": args.merge_vad,
    "merge_length_s": args.merge_length_s,
}


# Helper functions for OpenAI API response formatting
def format_srt_time(seconds: float) -> str:
    """將秒數轉換為 SRT 時間格式 (HH:MM:SS,mmm)"""
    if seconds is None:
        return "00:00:10,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """將秒數轉換為 VTT 時間格式 (HH:MM:SS.mmm)"""
    if seconds is None:
        return "00:00:10.000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_srt(text: str, duration: float = None) -> str:
    """格式化為 SRT 字幕格式"""
    start_time = "00:00:00,000"
    end_time = format_srt_time(duration) if duration else "00:00:10,000"
    srt = f"1\n{start_time} --> {end_time}\n{text}\n"
    return srt


def format_vtt(text: str, duration: float = None) -> str:
    """格式化為 WebVTT 字幕格式"""
    start_time = "00:00:00.000"
    end_time = format_vtt_time(duration) if duration else "00:00:10.000"
    vtt = f"WEBVTT\n\n1\n{start_time} --> {end_time}\n{text}\n"
    return vtt


def format_transcription_response(
    text: str, response_format: str = "json", duration: float = None, language: str = None
):
    """
    將辨識結果格式化為指定的回應格式
    
    Args:
        text: 辨識文本
        response_format: 回應格式 (json, text, verbose_json, srt, vtt)
        duration: 音頻時長（秒）
        language: 檢測到的語言
    
    Returns:
        格式化後的回應
    """
    if response_format == "text":
        return text
    elif response_format == "json":
        return {"text": text}
    elif response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "zh",
            "duration": duration,
            "text": text,
        }
    elif response_format == "srt":
        return format_srt(text, duration)
    elif response_format == "vtt":
        return format_vtt(text, duration)
    else:
        # 預設返回 json
        return {"text": text}



@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file")):
    suffix = audio.filename.split(".")[-1]
    audio_path = f"{args.temp_dir}/{str(uuid.uuid1())}.{suffix}"
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        logger.error(f"读取音频文件发生错误，错误信息：{e}")
        return {"msg": "读取音频文件发生错误", "code": 1}
    rec_results = model.generate(input=audio_bytes, is_final=True, **param_dict)
    
    # 檢查結果
    if not rec_results or len(rec_results) == 0:
        return {"text": "", "code": 0}
    
    rec_result = rec_results[0]
    if "text" not in rec_result or len(rec_result["text"]) == 0:
        return {"text": "", "code": 0}
    
    # 使用 rich_transcription_postprocess 處理結果
    text = rich_transcription_postprocess(rec_result["text"])
    
    # 簡化的回應格式
    ret = {"text": text, "code": 0}
    logger.info(f"識別結果：{ret}")
    return ret


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(..., description="audio file"),
    model_name: str = Form(..., alias="model", description="model to use"),
    language: Optional[str] = Form(None, description="language code (ISO-639-1)"),
    prompt: Optional[str] = Form(None, description="optional prompt"),
    response_format: Optional[str] = Form("json", description="response format"),
    temperature: Optional[float] = Form(0, description="sampling temperature"),
):
    """
    OpenAI-compatible audio transcription endpoint
    
    Compatible with OpenAI's /v1/audio/transcriptions API
    Supports response formats: json, text, verbose_json, srt, vtt
    """
    # 儲存上傳的文件
    suffix = file.filename.split(".")[-1] if "." in file.filename else "wav"
    audio_path = f"{args.temp_dir}/{str(uuid.uuid1())}.{suffix}"
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    
    try:
        # 使用 ffmpeg 轉換音頻
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        logger.error(f"音頻文件處理錯誤：{e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {"error": {"message": "Audio file processing failed", "type": "invalid_request_error"}}
    
    # 準備推理參數
    inference_params = {
        "language": language or args.language,
        "use_itn": args.use_itn,
        "merge_vad": args.merge_vad,
        "merge_length_s": args.merge_length_s,
    }
    
    # 執行辨識
    try:
        rec_results = model.generate(input=audio_bytes, is_final=True, **inference_params)
        
        if not rec_results or len(rec_results) == 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            formatted_response = format_transcription_response("", response_format, language=language)
            if response_format in ["text", "srt", "vtt"]:
                return Response(content=formatted_response, media_type="text/plain")
            return formatted_response
        
        rec_result = rec_results[0]
        if "text" not in rec_result or len(rec_result["text"]) == 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            formatted_response = format_transcription_response("", response_format, language=language)
            if response_format in ["text", "srt", "vtt"]:
                return Response(content=formatted_response, media_type="text/plain")
            return formatted_response
        
        # 後處理文本
        text = rich_transcription_postprocess(rec_result["text"])
        
        # 清理臨時文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # 格式化回應
        formatted_response = format_transcription_response(
            text=text,
            response_format=response_format,
            language=language
        )
        
        logger.info(f"OpenAI API 辨識結果：{text[:100]}...")
        
        # 根據回應格式設定 Content-Type
        if response_format == "text":
            return Response(content=formatted_response, media_type="text/plain")
        elif response_format in ["srt", "vtt"]:
            return Response(content=formatted_response, media_type="text/plain")
        else:
            return formatted_response
            
    except Exception as e:
        logger.error(f"辨識過程發生錯誤：{e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {"error": {"message": str(e), "type": "server_error"}}



if __name__ == "__main__":
    uvicorn.run(
        app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile
    )
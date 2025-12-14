#!/usr/bin/env python3
"""
Speed test script for OpenAI API compatible transcription endpoint
"""

import argparse
import json
import subprocess
import time
import requests
from pathlib import Path


def get_audio_duration(audio_file: str) -> float:
    """
    Get audio duration in seconds using ffprobe
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Duration in seconds, or None if unable to get duration
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return None


def test_transcription_speed(
    audio_file: str,
    api_url: str = "http://localhost:8000/v1/audio/transcriptions",
    model: str = "sensevoice",
    response_format: str = "json",
    language: str = None,
    num_runs: int = 1
):
    """
    Test the transcription API speed
    
    Args:
        audio_file: Path to the audio file
        api_url: API endpoint URL
        model: Model name
        response_format: Response format (json, text, verbose_json, srt, vtt)
        language: Language code (optional)
        num_runs: Number of test runs
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Get file size and duration
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    audio_duration = get_audio_duration(audio_file)
    
    print(f"Testing transcription speed")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    if audio_duration:
        print(f"Audio duration: {audio_duration:.2f}s")
    print(f"API URL: {api_url}")
    print(f"Model: {model}")
    print(f"Response format: {response_format}")
    print(f"Language: {language or 'auto'}")
    print(f"Number of runs: {num_runs}")
    print(f"{'='*60}\n")
    
    times = []
    results = []
    
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")
        
        # Prepare request
        with open(audio_file, "rb") as f:
            files = {"file": f}
            data = {
                "model": model,
                "response_format": response_format
            }
            if language:
                data["language"] = language
            
            # Measure time
            start_time = time.time()
            
            try:
                response = requests.post(api_url, files=files, data=data)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if response.status_code == 200:
                    times.append(elapsed_time)
                    
                    # Parse result based on format
                    if response_format in ["json", "verbose_json"]:
                        result = response.json()
                        text = result.get("text", "")
                    else:
                        text = response.text
                    
                    results.append(text)
                    
                    print(f"  ✓ Time: {elapsed_time:.3f}s")
                    if i == 0:  # Only print result for first run
                        print(f"  Result: {text[:100]}{'...' if len(text) > 100 else ''}")
                else:
                    print(f"  ✗ Error: HTTP {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
            
            except Exception as e:
                print(f"  ✗ Exception: {e}")
        
        print()
    
    # Print statistics
    if times:
        print(f"\n{'='*60}")
        print(f"STATISTICS")
        print(f"{'='*60}")
        if audio_duration:
            print(f"Audio duration: {audio_duration:.2f}s")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Total runs: {len(times)}")
        print(f"Successful runs: {len(times)}")
        print()
        print(f"Average time: {sum(times) / len(times):.3f}s")
        print(f"Min time: {min(times):.3f}s")
        print(f"Max time: {max(times):.3f}s")
        
        if len(times) > 1:
            import statistics
            print(f"Std deviation: {statistics.stdev(times):.3f}s")
        
        print()
        print(f"Throughput: {file_size_mb / (sum(times) / len(times)):.2f} MB/s")
        
        if audio_duration:
            avg_time = sum(times) / len(times)
            rtf = avg_time / audio_duration
            print(f"RTF (Real-Time Factor): {rtf:.2f}x")
            print(f"  (Process time / Audio duration: {avg_time:.3f}s / {audio_duration:.2f}s)")
        
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAI API compatible transcription speed"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/audio/transcriptions",
        help="API endpoint URL (default: http://localhost:8000/v1/audio/transcriptions)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sensevoice",
        help="Model name (default: sensevoice)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "text", "verbose_json", "srt", "vtt"],
        help="Response format (default: json)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., zh, en, yue)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of test runs (default: 1)"
    )
    
    args = parser.parse_args()
    
    test_transcription_speed(
        audio_file=args.audio_file,
        api_url=args.url,
        model=args.model,
        response_format=args.format,
        language=args.language,
        num_runs=args.runs
    )


if __name__ == "__main__":
    main()

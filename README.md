# Baberu: Automated Subtitle Generation and Processing Pipeline

This is a hobbyist project. No issues or tickets will be responded to. Code is subject to change without notice.

## Overview

Baberu is a command-line utility for automating subtitle generation, refinement, and translation. It processes media URLs, local video/audio files, or existing transcripts/subtitles through a configurable pipeline. Using external APIs like ElevenLabs and Google Gemini, it produces refined subtitles, typically in `.ass` format, handling steps from download and transcription to timing correction and translation.

## Features

*   **Versatile Input:** Accepts media URLs, local video/audio files, ElevenLabs JSON transcripts, or existing subtitle files (`.srt`, `.ass`, `.vtt`).
*   **Modular Workflow:** Activate specific processing stages (download, extract, transcribe, convert, fix, translate) via command-line flags.
*   **Automated Audio Extraction:** Extracts audio (defaults to `.opus`) from video sources.
*   **AI-Powered Transcription:** Utilizes ElevenLabs API for speech-to-text.
*   **Subtitle Generation:** Converts transcriptions into timed subtitles (default: `.ass`).
*   **Accuracy Enhancement (Two-Pass):** Optionally re-transcribes low-confidence segments for improved accuracy.
*   **Automated Timing Correction:** Applies heuristics to fix common subtitle timing issues.
*   **Contextual Translation:** Translates subtitles using Google Gemini, leveraging auto-generated or provided context. Supports partial progress saving.
*   **Efficient Processing:** Skips completed stages by detecting existing intermediate files (e.g., `.opus`, `.json`, `.raw.ass`, `.en.ass`), enabling easy pipeline resumption.
*   **Output Customization:** Allows specification of output directories and final filenames.

## Prerequisites

1.  **Python:** Python 3.8 or higher recommended.
2.  **FFmpeg:** Required for audio extraction and potential media processing. Ensure `ffmpeg` is installed and accessible in your system's PATH.
3.  **API Keys:**
    *   **ElevenLabs API Key:** Required for the transcription step (`--speech-to-text`).
    *   **Google Gemini API Key:** Required for context generation and translation (`--translate`).
    *   It is recommended to set these as environment variables:
        ```bash
        export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
        export GOOGLE_API_KEY="your_google_gemini_api_key"
        ```

## Installation

1.  **Set up prerequisites:** Ensure FFmpeg, and API keys are set up as described above.
2.  **Download and install baberu:**
    ```bash
    git clone https://github.com/ui3TD/baberu
    cd baberu
    pip install .
    ```

## Usage

Execute the script from the command line, providing the source input and flags to enable specific processing stages.

**Syntax:**

```bash
baberu <source_file_path_or_url> [options]
```

**Source Input:**

*   `<source_file_path_or_url>`: Required. Can be a URL (e.g., YouTube), a local video/audio file, an ElevenLabs `.json` transcription, or a subtitle file (`.srt`, `.ass`, `.vtt`).

**Core Options (Pipeline Stages):**

Specify flags to activate corresponding steps. Steps are skipped if their expected output file already exists.

*   `--extract`: Extract audio (to `.opus`) from video input.
*   `--speech-to-text`: Transcribe audio to text (`.json`) using ElevenLabs. (Requires API key).
*   `--convert`: Convert `.json` transcription to raw subtitles (`.raw.ass`).
*   `--retranscribe [PATH|'auto']`: Refine specific subtitle segments via re-transcription (generates `.2pass.ass`). Requires audio: `'auto'` uses pipeline audio, `PATH` specifies an audio file. (Requires ElevenLabs API key).
*   `--fix`: Apply automated timing corrections (generates `.fixed.ass`).
*   `--translate [PATH|'auto']`: Translate subtitles to English (generates `.en.ass`) using Google Gemini. (Requires API key).
    *   `'auto'`: Generate/use context summary (`.context.txt`).
    *   `PATH`: Use context from the specified text file (cannot be named `auto`).

**Output Control:**

*   `-o PATH`, `--output PATH`: Specify the final output file path (e.g., the final `.ass` file).
*   `-d PATH`, `--directory PATH`: Specify the directory for all output files. Defaults to input file's directory or current directory for URLs.

**Help:**

*   `-h`, `--help`: Display detailed help message and exit.

**Examples:**

1.  **Full pipeline from URL (download, extract, transcribe, convert, fix, translate):**
    ```bash
    baberu "youtube_url" --extract --speech-to-text --convert --fix --translate auto -d ./output
    ```

2.  **Translate existing subtitles with custom context:**
    ```bash
    baberu ./input.ass --translate ./my_context.txt -o ./input.en.ass
    ```

3.  **Transcribe local audio and convert:**
    ```bash
    baberu ./meeting.mp3 --speech-to-text --convert -o ./meeting.raw.ass
    ```

4.  **Apply two-pass refinement and fixes to existing subs, using pipeline audio:**
    ```bash
    # Assumes video.mp4 and corresponding video.raw.ass exist
    baberu ./video.raw.ass --retranscribe auto --fix -o ./video.fixed.ass -d ./subs_dir
    ```

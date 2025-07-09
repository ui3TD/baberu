# baberu: Automated Subtitle Generation and Processing Pipeline

This is a hobbyist project. No contributions will be accepted. Features are subject to change without notice.

## Overview

"baberu" is a command-line utility for automating subtitle generation, refinement, and translation. It processes media URLs, local video/audio files, or existing transcripts/subtitles through a configurable pipeline. Using external APIs like ElevenLabs and Google Gemini, it produces refined subtitles, typically in `.ass` format, handling steps from download and transcription to timing correction and translation.

## Features

*   **Flexible Input:** Handles media URLs, local video/audio files, JSON transcripts, and existing subtitle files (`.srt`, `.ass`, `.vtt`).
*   **Configurable Pipeline:** Run a full workflow with `--auto-pilot` or execute specific stages individually. The pipeline is resumable, automatically skipping completed steps.
    *   **`extract`:** Pulls audio from video files.
    *   **`speech-to-text`:** Transcribes audio to a text file using an external API.
    *   **`convert`:** Generates timed `.ass` subtitles from a transcript, leveraging an LLM to break long lines.
    *   **`retranscribe`:** Improves accuracy by re-processing low-confidence segments (two-pass).
    *   **`fix`:** Applies automated corrections for common timing issues.
    *   **`translate`:** Translates subtitles using an LLM with user-provided or auto-generated grounded context.
    *   **`pad`:** Adjusts subtitle timings to meet readability standards (CPS, lead-in/out).
*   **AI Service Integration:**
    *   **Transcription:** ElevenLabs, OpenAI (Whisper), Fireworks AI (Whisper-v3).
    *   **Translation (LLMs):** Google Gemini, Anthropic Claude, OpenAI GPT, Grok, Deepseek, and various models via OpenRouter.
*   **Targeted Processing:** Apply fixes, translations, or re-transcription to a specific range of lines with the `--lines` flag.
*   **Output Options:**
    *   Generate a final subtitles file in any common format (`--output`).
    *   Create a video from an audio file and a static image (`--audio-to-video`).
    *   Burn subtitles directly onto a video (`--hardcode`).

## Prerequisites

1.  **Python:** Python 3.8 or higher recommended.
2.  **FFmpeg:** Required for audio extraction and potential media processing. Ensure `ffmpeg` is installed and accessible in your system's PATH.
3.  **API Keys:**
    `baberu` integrates with various AI services. You only need to set up the keys for the services you intend to use. It is recommended to set these as environment variables.

    **Transcription Providers (for `--speech-to-text` and `--retranscribe`):**
    *   **ElevenLabs:** For models containing `scribe`.
        ```bash
        export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
        ```
    *   **OpenAI:** For models containing `whisper-1`.
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
        ```
    *   **Fireworks AI:** For models containing `whisper-v3`.
        ```bash
        export FIREWORKS_API_KEY="your_fireworks_ai_api_key"
        ```

    **Language Model (LLM) Providers (for `--translate`, optionally `--convert` and `--retranscribe`):**
    *   **Google Gemini:** For models containing `gemini`.
        ```bash
        export GEMINI_API_KEY="your_google_api_key"
        ```
    *   **OpenAI:** For models containing `gpt`, `o1`, `o3`, or `o4`.
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
        ```
    *   **Anthropic:** For models containing `claude`.
        ```bash
        export ANTHROPIC_API_KEY="your_anthropic_api_key"
        ```
    *   **xAI:** For models containing `grok`.
        ```bash
        export XAI_API_KEY="your_xai_api_key"
        ```
    *   **Deepseek:** For models containing `deepseek`.
        ```bash
        export DEEP_API_KEY="your_deepseek_api_key"
        ```
    *   **OpenRouter:** For any model specified in `creator/model` format (e.g., `mistralai/mistral-7b-instruct`).
        ```bash
        export OPENROUTER_API_KEY="your_openrouter_api_key"
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

**Workflow Control**

*   `-A`, `--auto-pilot`: Activates a full, standard pipeline: extract → speech-to-text → convert → retranscribe → fix → translate → pad. This is the most convenient option for a complete, one-shot workflow.

**Core Options (Pipeline Stages):**

Specify flags to activate corresponding steps. Steps are skipped if their expected output file already exists.

*   `-x`, `--extract`: Extract audio (to `.opus`) from video input.
*   `-s`, `--speech-to-text`: Transcribe audio to text (`.json`) using ElevenLabs. (Requires API key).
*   `-c`, `--convert`: Convert `.json` transcription to raw subtitles (`.raw.ass`).
*   `-r [PATH|'auto']`, `--retranscribe [PATH|'auto']`: Refine specific subtitle segments via re-transcription (generates `.2pass.ass`). Requires audio: `'auto'` uses pipeline audio, `PATH` specifies an audio file. (Requires ElevenLabs API key).
*   `-f`, `--fix`: Apply automated timing corrections (generates `.fixed.ass`).
*   `-t [PATH|'auto']`, `--translate [PATH|'auto']`: Translate subtitles to English (generates `.en.ass`) using Google Gemini. (Requires API key).
    *   `'auto'`: Generate/use context summary (`.context.txt`).
    *   `PATH`: Use context from the specified text file (cannot be named `auto`).
*   `-p`, `--pad`: Apply timing padding and conform to readability standards (generates `.padded.ass`).

**Targeted Processing**

*   `--lines XX-YY|XX`: Restrict processing for `retranscribe`, `fix`, `translate`, and `pad` stages to a specific line range (e.g., `10-15`) or a single line (e.g., `10`). Indices are 1-based and inclusive. This generates a separate output file (e.g., `.fixed_custom.ass`).

**Output & Finalization**

*   `-o PATH`, `--output PATH`: Specify the final output file path (e.g., the final `.ass` or hardcoded `.mp4` file).
*   `-d PATH`, `--directory PATH`: Specify the directory for all intermediate and final output files. Defaults to the input file's directory or the current directory for URLs.
*   `--hardcode VIDEO_PATH`: Burn the resulting subtitles onto a video. Provide the path to the target video file.
*   `--audio-to-video IMAGE_PATH`: Create a video by combining the source audio file with a specified static image.

**Help:**

*   `-h`, `--help`: Display detailed help message and exit.

**Examples:**

1.  **Full pipeline using Auto-Pilot:**
    ```bash
    # This single command runs the entire standard pipeline on a video URL
    baberu "youtube_url" -A -o ./final_subs.ass -d ./output
    ```

2.  **Translate existing subtitles with custom context and then pad them:**
    ```bash
    baberu ./input.ass -t ./my_context.txt -p -o ./input.en.padded.ass
    ```

3.  **Transcribe local audio and convert:**
    ```bash
    baberu ./meeting.mp3 -s -c -o ./meeting.raw.ass
    ```

4.  **Apply two-pass refinement and fixes to existing subs, using pipeline audio:**
    ```bash
    # Assumes video.mp4 and corresponding video.raw.ass exist
    baberu ./video.raw.ass -r auto -f -o ./video.fixed.ass -d ./subs_dir
    ```

5.  **Fix only lines 50 to 62 in an existing subtitle file:**
    ```bash
    # Generates a .fixed_custom.ass file
    baberu subs.ass -f --lines 50-62
    ```

6.  **Generate subtitles and burn them into the source video:**
    ```bash
    # Runs the full pipeline and then hardcodes the result onto the original video
    baberu ./my_video.mp4 -A --hardcode ./my_video.mp4 -o ./my_video.hardcoded.mp4
    ```
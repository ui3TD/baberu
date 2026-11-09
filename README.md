# baberu: Automated Subtitle Generation and Processing Pipeline

This is a hobbyist project. No contributions will be accepted. Features are subject to change without notice.

## Overview

"baberu" is a command-line tool for generating and translating subtitles from videos, audio files, or existing transcripts. It automates steps like audio extraction, speech-to-text transcription, subtitle timing, corrections for readability, translation, and padding, using AI services such as ElevenLabs for transcription and models like Gemini or GPT for translation. Use it to create accurate subtitles quickly, with options for full pipelines or targeted fixes, and output in formats like ASS or hardcoded videos.

## Features

*   **Flexible Input:** Handles media URLs, local video/audio files, JSON transcripts, and existing subtitle files (`.srt`, `.ass`, `.vtt`).
*   **Configurable Pipeline:** Run everything by default, or name specific stages. Work is saved as it goes, so an interrupted run picks up where it stopped.
    *   **`extract`:** Pulls audio from video files.
    *   **`speech-to-text`:** Transcribes audio to a text file using an external API.
    *   **`convert`:** Generates timed `.ass` subtitles from a transcript, leveraging an LLM to break long lines.
    *   **`retranscribe`:** Improves accuracy by re-processing low-confidence segments (two-pass).
    *   **`fix`:** Applies automated corrections for common timing issues.
    *   **`translate`:** Translates subtitles using an LLM with user-provided or auto-generated grounded context.
    *   **`pad`:** Adjusts subtitle timings to meet readability standards (CPS, lead-in/out).
*   **AI Service Integration:**
    *   **Transcription:** ElevenLabs, OpenAI (Whisper).
    *   **Translation (LLMs):** Google Gemini, Anthropic Claude, OpenAI GPT, Grok, and various models via OpenRouter.
*   **Targeted Processing:** Apply fixes, translations, or re-transcription to a specific range of lines with the `--lines` flag.
*   **Output Options:**
    *   Generate a final subtitles file in any common format (`--output`).
    *   Create a video from an audio file and a static image (`--audio-to-video`).
    *   Burn subtitles directly onto a video (`--hardcode`).

## Prerequisites

1.  **Python:** Python 3.12 or higher.
2.  **FFmpeg:** Required for audio extraction and potential media processing. Ensure `ffmpeg` is installed and accessible in your system's PATH.
3.  **API Keys:**
    `baberu` integrates with various AI services. You only need to set up the keys for the services you intend to use. It is required to set these as environment variables.

    **Transcription Providers (for `--speech-to-text` and `--retranscribe`):**
    *   **ElevenLabs:** For models containing `scribe`.
        ```bash
        export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
        ```
    *   **OpenAI:** For models containing `whisper-1`.
        ```bash
        export OPENAI_API_KEY="your_openai_api_key"
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

```bash
baberu <url-or-file> [options]
```

Give baberu a URL, video, audio, `.json` transcript or subtitle file. With no stage options it runs every stage that applies and leaves one result beside the source:

```
videos/
  show.mp4
  show.en.ass      the result; video players load this name automatically
  show.baberu/     everything used to make it
```

The result's path is printed on stdout and progress on stderr. Use `--dry-run` to see what a command would do and which stages cost money.

Keep the `.baberu` folder if you might change anything later, since it holds the paid transcription and translation. `--clean` deletes it after a successful run.

### Stages

Naming a stage runs that stage and anything it depends on, so `baberu show.mp4 -t` extracts, transcribes and converts first. `retranscribe`, `fix` and `pad` run only when named or on a full run.

| Option | Stage |
|---|---|
| `-A`, `--auto-pilot` | Every stage (the default when none is named) |
| `-x`, `--extract` | Extract the audio from a video |
| `-s`, `--speech-to-text` | Transcribe the audio (paid) |
| `-c`, `--convert` | Turn a transcript into subtitles |
| `-r`, `--retranscribe` | Transcribe badly timed segments again (paid) |
| `-f`, `--fix` | Fix subtitle timing |
| `-t`, `--translate` | Translate (paid) |
| `-p`, `--pad` | Extend timing for readability |
| `--hardcode` | Burn subtitles into the video |
| `--audio-to-video IMAGE` | Make a video from the audio and an image |

### Additional inputs

Each of these turns on the stage that uses it. Audio and video with the same name as the source are found automatically.

| Option | Use |
|---|---|
| `--audio FILE` | Audio for retranscribing |
| `--context FILE` | Background notes for translating, instead of generating them |
| `--subs FILE` | Subtitles to hardcode |
| `--video FILE` | Video to hardcode into |

### Output

| Option | Effect |
|---|---|
| `-o PATH` | Where the final result goes: a file or a folder. Its type must suit the result |
| `-d DIR` | Folder for the result and the work folder (default: beside the source) |
| `--overwrite` | Replace an existing file baberu didn't write |
| `--clean` | Delete the work folder after a successful run |

baberu never overwrites its input, and it won't replace a file it didn't write, including a result you've since edited, without `--overwrite`.

### Reusing and redoing work

A second run reuses earlier work and says so. Naming a file means starting from it. `fix` and `pad` always run again because they're free.

If paid work no longer matches, because the subtitles or model changed, baberu stops before spending anything and tells you to redo it:

```bash
baberu show.mp4 --redo translate
```

`--redo` accepts `extract`, `transcribe`, `convert`, `retranscribe`, `context`, `translate` or `all`, and can be repeated.

### Other options

| Option | Effect |
|---|---|
| `--model NAME` | Translation model |
| `--lang-from CODE`, `--lang-to CODE` | Source and target language |
| `--lines 5-20` | Work on only these lines, leaving the rest unchanged |
| `--dry-run` | Show the plan without doing or spending anything |
| `--config FILE` | Use this config file |
| `--debug` | Show detailed logs, and a traceback on errors |

### Examples

```bash
# Everything, from a URL, with results in ./output
baberu "https://youtube.com/watch?v=..." -d ./output

# Translate existing subtitles with your own notes, then pad them
baberu input.ass --context notes.txt -p -o input.en.ass

# Transcribe and stop at subtitles
baberu meeting.mp3 -c

# Retranscribe and fix subtitles, finding video.mp4 beside them
baberu video.ass -r -f

# Fix only lines 50 to 62
baberu subs.ass -f --lines 50-62

# Full run, then burn the result into the source video
baberu my_video.mp4 -A --hardcode -o my_video.subbed.mp4

# Try another translation model
baberu show.mp4 --model claude-opus-5-5 --redo translate
```

### Upgrading from earlier versions

Work from older versions (`show.raw.ass`, `show.json` and so on) is copied into the new work folder the first time you run that job, and the original files are left alone.

Some options changed: `-t auto` is now `-t`, `-t FILE` is `--context FILE`, `-r FILE` is `--audio FILE`, and `--hardcode FILE` is `--hardcode` with `--subs` or `--video` if needed.

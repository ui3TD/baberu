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
    *   **`retranscribe`:** Transcribes again the stretches whose timing looks wrong, and keeps the second pass only where it is better (two-pass).
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

### Workflows

**Run everything.** `baberu URL` downloads, transcribes, fixes timing, translates and pads. Add `--nekocap` to upload the result when it's done. Add `-y` when nobody is at the keyboard.

**Review as you go.** baberu pauses after writing `keyterms.txt` (before transcribing) and `context.txt` (before translating) so you can edit them. You can also stop at any stage (`-s`, `-c`, `-f`), edit what's in the work folder, and run again: finished work is reused.

**Correct the result.** Two ways:
- Edit `show.en.ass` by hand. baberu keeps your edits, and a later run that has nothing new for the file leaves it alone. A run that would change the file stops first; use `--overwrite` to replace it.
- Translate some lines again: `baberu show.mp4 -t --lines 67-68`. Only those lines change, inside the job's translation, and the result is rebuilt. The line numbers are the result's.

**Publish.** `baberu publish URL` (or the video, or `show.en.ass`) uploads the result exactly as it is, hand edits included, and runs nothing else. See [Uploading to NekoCap](#uploading-to-nekocap).

### Stages

Naming a stage runs that stage and anything it depends on, so `baberu show.mp4 -t` extracts, transcribes and converts first. A job keeps its own chain: once a full run has made it, `-t` alone still translates the retranscribed, fixed subtitles and pads the result, reusing all of that. On a new job, `retranscribe`, `fix` and `pad` run only when named or on a full run.

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

baberu never overwrites its input. It won't replace a file it didn't write, including a result you've since edited, without `--overwrite`; when it has nothing new for an edited result, it keeps your version.

### Reusing and redoing work

A second run reuses earlier work and says so, `fix` and `pad` included when their input and settings haven't changed. Naming a file means starting from it. A URL whose download is already in the folder is planned from that download, so `--dry-run` shows the whole plan.

If paid work no longer matches, because the subtitles or model changed, baberu stops before spending anything (a dry run reports it too) and tells you to redo it:

```bash
baberu show.mp4 --redo translate
```

`--redo` accepts `extract`, `keyterms`, `transcribe`, `convert`, `retranscribe`, `context`, `translate` or `all`, and can be repeated.

### Other options

| Option | Effect |
|---|---|
| `--model NAME` | Translation model |
| `--lang-from CODE`, `--lang-to CODE` | Source and target language |
| `--lines 5-20` | Work on only these lines, leaving the rest unchanged. With `-t` on a job that has a translation, those lines are translated again in place; otherwise the result is a separate `.L5-20` file |
| `--keyterms FILE` | Add the terms in this keyterms file to the channel's |
| `--no-keyterms` | Send no keyterms to the transcription service |
| `--dry-run` | Show the plan without doing or spending anything |
| `-y`, `--yes` | Don't pause to review the generated keyterms or context. Without a keyboard, baberu carries on by itself anyway |
| `--diagnose` | Report a transcript's timing problems and save `health.json` in the work folder (free) |
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

### Keyterms

Scribe models take a list of names and jargon to spell right. Each YouTube channel has one file holding every term for it, named in the config:

```yaml
keyterms:
  channels:
    UCxxxxxxxxxxxxxxxxxxxxxx: mychannel   # keyterms/mychannel.yaml
```

```yaml
# keyterms/mychannel.yaml
channel_id: UCxxxxxxxxxxxxxxxxxxxxxx
channel: My Channel
terms:
  - Jane Doe
  - The Weekly Show
```

The file holds only `channel_id`, `channel` and `terms`, a plain list of strings. Anything else is an error. A video from a channel with no file stops before transcribing and names the line and file to add. Set `require_channel_file: false` to go on with only the video's own details.

Each job writes the terms it sends to `keyterms.txt`: a model's picks from the title and description, then the channel's terms, then the video's tags. If the channel's file changes, the next transcription makes `keyterms.txt` again. `--redo keyterms` does that by hand.

### Comparing transcription models

Every run saves `health.json` in the work folder: a profile of the transcript (token sizes, durations, gaps, speakers, whether confidence varies), its anomalies with times, and how often each repair fired. The anomalies are:

| Kind | Meaning |
|---|---|
| `smear` | A token held far longer than its letters could take to say |
| `collapse` | Several words squeezed into almost no time |
| `loop` | The same few tokens repeated (numbers never count) |
| `disorder` | Missing, backwards or overlapping timestamps |
| `low_confidence` | Unsure words, only when the model's confidence varies at all |

`--diagnose` prints the same report for a transcript without spending anything. To compare models, transcribe the same file into separate folders and compare the two reports:

```bash
baberu show.mp4 -s -d v1 --config config.v1.yaml
baberu show.mp4 -s -d v3 --config config.v3.yaml
baberu v3/show.baberu/transcript.json --diagnose
```

Conversion fixes smears and collapses for free; `retranscribe` sends whatever it couldn't fix, and runs of very short lines, to be transcribed again. Limits live under `anomalies:` in the config, request parameters under `transcription.options`, and settings for one model or language under `profiles:`. A job remembers which model and options made its transcript and stops if they change; use `--redo transcribe`.

### Uploading to NekoCap

`baberu publish` uploads the finished translation of a YouTube video to your [NekoCap](https://nekocap.com) account, exactly as the file is on disk. It runs no stages and changes no files. Give it the video's URL, the downloaded video, or the result itself:

```bash
baberu publish "https://www.youtube.com/watch?v=..." --dry-run   # show the file, privacy and title; check the login
baberu publish show.en.ass --unlisted
baberu publish show.en.ass --title "Mirinya, First Time at Hidakaya"
```

The title is translated with the translation model and the job's `context.txt`, so names are spelled as in the subtitles. It is kept for later uploads: `--title` sets it yourself, `--retitle` translates it again. Uploads are public by default (`nekocap.privacy` in the config); `--unlisted` or `--public` override that. Publishing again replaces your earlier caption for that video and language, and does nothing if the file, title and privacy haven't changed.

`baberu URL --nekocap` (or `--nekocap-unlisted`) does a full run, or the stages you name, then publishes the same way.

baberu uses your browser's NekoCap login. Run `baberu-nekocap-login`: it opens nekocap.com and puts a line on your clipboard. Sign in if needed, press F12, paste it into the Console and press Enter. Then press Enter in the terminal. It checks the login and saves it to `.env` as `NEKOCAP_SESSION_TOKEN`. When the login stops working, baberu says so; run it again. A dry run checks the login without uploading.

### Upgrading from earlier versions

Work from older versions (`show.raw.ass`, `show.json` and so on) is copied into the new work folder the first time you run that job, and the original files are left alone.

Some options changed: `-t auto` is now `-t`, `-t FILE` is `--context FILE`, `-r FILE` is `--audio FILE`, and `--hardcode FILE` is `--hardcode` with `--subs` or `--video` if needed.

Keyterm lists (`keyterms/<name>.txt`, several per channel) are now one `keyterms/<name>.yaml` per channel, and `keyterms.channels` maps each channel to one name. `lists_dir` is now `dir`. See [Keyterms](#keyterms).

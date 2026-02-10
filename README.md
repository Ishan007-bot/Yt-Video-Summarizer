# YouTube Video Summarizer

A Streamlit web app that takes a YouTube video URL, transcribes its audio using **OpenAI Whisper**, and generates a concise summary using a **HuggingFace Transformers** summarization model. Runs entirely on CPU -- no GPU or API keys required.

## How It Works

1. **Download** -- the audio track of a YouTube video is downloaded via `pytubefix`.
2. **Transcribe** -- OpenAI Whisper (through Haystack's `WhisperTranscriber`) converts speech to text.
3. **Summarize** -- a HuggingFace summarization pipeline (`sshleifer/distilbart-cnn-12-6`) condenses the transcript into a short summary.
4. **Display** -- the video and its summary are shown side-by-side in a Streamlit UI.

## Tech Stack

| Component | Library |
|-----------|---------|
| Web UI | [Streamlit](https://streamlit.io/) |
| NLP Pipeline | [Haystack](https://haystack.deepset.ai/) (v1.x) |
| Speech-to-Text | [OpenAI Whisper](https://github.com/openai/whisper) |
| Summarization | [HuggingFace Transformers](https://huggingface.co/sshleifer/distilbart-cnn-12-6) |
| Video Download | [pytubefix](https://github.com/JuanBindez/pytubefix) |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Ishan007-bot/Yt-Video-Summarizer.git
cd Yt-Video-Summarizer

# Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# Run the app
streamlit run yt_summary.py
```

Open **http://localhost:8501**, paste a YouTube URL, and click **Submit**.

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `SUMMARY_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarization model |
| `YT_DOWNLOAD_DIR` | `/tmp/yt_downloads` | Directory for downloaded audio files |

## Project Structure

```
.
├── yt_summary.py      # Main Streamlit app
├── model_add.py       # Custom Haystack invocation layer (HF summarization)
├── summary.py         # Standalone CLI summarization script
├── requirements.txt   # Python dependencies
├── .streamlit/        # Streamlit config (headless, no email prompt)
├── LICENSE
└── README.md
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**If you like this project, drop a star to the repo!**

import os
import time

# Disable Haystack telemetry before importing (avoids PermissionError on .haystack dir)
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

import streamlit as st
from pytubefix import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer

# Add ffmpeg to PATH for Whisper
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

st.set_page_config(
    layout="wide"
)

import tempfile

# Download directory -- uses /tmp by default (works on Linux & HF Spaces);
# override with YT_DOWNLOAD_DIR env var for local Windows usage.
DOWNLOAD_DIR = os.environ.get("YT_DOWNLOAD_DIR", os.path.join(tempfile.gettempdir(), "yt_downloads"))

def download_video(url):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download(output_path=DOWNLOAD_DIR)

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    # "base" = faster; set env WHISPER_MODEL=tiny for max speed, or small/medium for better accuracy
    whisper_model = os.environ.get("WHISPER_MODEL", "base")
    whisper = WhisperTranscriber(model_name_or_path=whisper_model)
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():

    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()
        try:
            with st.spinner("Downloading video..."):
                file_path = download_video(youtube_url)
            st.success("Downloaded.")

            with st.spinner("Loading summarization model (first time may take a minute)..."):
                model_path = os.environ.get("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
                model = initialize_model(model_path)
                prompt_node = initialize_prompt_node(model)
            st.success("Model ready.")

            with st.spinner("Transcribing audio with Whisper (1–5 min depending on video length)..."):
                output = transcribe_audio(file_path, prompt_node)
            st.success("Done transcribing.")

            end_time = time.time()
            elapsed_time = end_time - start_time

            col1, col2 = st.columns([1, 1])
            with col1:
                st.video(youtube_url)
            with col2:
                st.header("Summarization of YouTube Video")
                summary_text = output["results"][0] if output.get("results") else str(output)
                st.success(summary_text)
                st.write(f"Time taken: {elapsed_time:.2f} seconds")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

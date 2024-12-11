# app.py

# -*- coding: utf-8 -*-
"""
Vietnamese End-to-End Speech Recognition using Wav2Vec 2.0 with Speaker Diarization.
Streamlit Application.
"""

import os
import zipfile
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from huggingface_hub import hf_hub_download
import streamlit as st
import numpy as np
import librosa
# from pyAudioAnalysis import audioSegmentation as aS  # Consider alternative if issues persist

# Optional: Use logging for better debugging
import logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(cache_dir='./cache/'):
    st.info("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )
    
    st.info("Downloading language model...")
    lm_zip_file = hf_hub_download(
        repo_id="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        filename="vi_lm_4grams.bin.zip",
        cache_dir=cache_dir
    )
    
    st.info("Extracting language model...")
    with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    
    lm_file = os.path.join(cache_dir, 'vi_lm_4grams.bin')
    if not os.path.isfile(lm_file):
        raise FileNotFoundError(f"Language model file not found: {lm_file}")
    
    st.success("Processor, model, and language model loaded successfully.")
    return processor, model, lm_file

@st.cache_resource(show_spinner=False)
def get_decoder_ngram_model(_tokenizer, ngram_lm_path):
    st.info("Building decoder with n-gram language model...")
    vocab_dict = _tokenizer.get_vocab()
    sorted_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab_list = [token for _, token in sorted_vocab][:-2]  # Exclude special tokens
    
    alphabet = Alphabet.build_alphabet(vocab_list)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    st.success("Decoder built successfully.")
    return decoder

def transcribe_chunk(model, processor, decoder, speech_chunk, sampling_rate):
    if speech_chunk.ndim > 1:
        speech_chunk = np.mean(speech_chunk, axis=1)
    speech_chunk = speech_chunk.astype(np.float32)

    target_sr = 16000
    if sampling_rate != target_sr:
        speech_chunk = librosa.resample(speech_chunk, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr

    # Define minimum duration (e.g., 0.5 seconds)
    MIN_DURATION = 0.5  # seconds
    MIN_SAMPLES = int(MIN_DURATION * sampling_rate)

    if len(speech_chunk) < MIN_SAMPLES:
        # Pad with zeros (silence) to reach minimum length
        padding = MIN_SAMPLES - len(speech_chunk)
        speech_chunk = np.pad(speech_chunk, (0, padding), 'constant')

    input_values = processor(
        speech_chunk, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits[0]

    beam_search_output = decoder.decode(
        logits.cpu().detach().numpy(),
        beam_width=500
    )
    return beam_search_output

def alternative_speaker_diarization(audio_file, num_speakers=2):
    """
    Alternative speaker diarization method with more robust error handling
    """
    try:
        # Use librosa to load the audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Rough segmentation based on energy
        intervals = librosa.effects.split(y, top_db=30)  # Adjust top_db as needed

        # Merge very short intervals
        MIN_INTERVAL_DURATION = 0.5  # seconds
        MIN_SAMPLES = int(MIN_INTERVAL_DURATION * sr)
        merged_intervals = []
        for interval in intervals:
            if merged_intervals and (interval[0] - merged_intervals[-1][1]) < MIN_SAMPLES:
                # Merge with the previous interval
                merged_intervals[-1][1] = interval[1]
            else:
                merged_intervals.append([interval[0], interval[1]])

        # Assign speakers cyclically
        segments = []
        for i, (start, end) in enumerate(merged_intervals):
            speaker_id = i % num_speakers
            start_time = start / sr
            end_time = end / sr
            segments.append((start_time, end_time, speaker_id))

        return segments

    except Exception as e:
        st.error(f"Speaker diarization failed: {e}")
        # Fallback to a simple equal-length segmentation
        audio, sr = sf.read(audio_file)
        total_duration = len(audio) / sr
        segment_duration = total_duration / num_speakers

        segments = []
        for i in range(num_speakers):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            segments.append((start, end, i))

        return segments

def process_segments(audio_file, segments, model, processor, decoder, sampling_rate=16000):
    speech, sr = sf.read(audio_file)
    final_transcriptions = []

    # Remove duplicate or overlapping segments
    unique_segments = []
    for segment in sorted(segments, key=lambda x: x[0]):
        if not unique_segments or segment[0] >= unique_segments[-1][1]:
            unique_segments.append(segment)

    for start, end, speaker_id in unique_segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        speech_chunk = speech[start_sample:end_sample]
        transcript = transcribe_chunk(model, processor, decoder, speech_chunk, sr)

        # Only add non-empty transcripts
        if transcript.strip():
            final_transcriptions.append((f"Speaker {speaker_id + 1}", transcript))

    return final_transcriptions

def main():
    st.title("🇻🇳 Vietnamese Speech Recognition with Speaker Diarization")

    st.write("""
    Upload an audio file, select the number of speakers, and get the transcribed text along with speaker labels.
    """)

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3", "flac", "m4a"])
    num_speakers = st.sidebar.slider("Number of Speakers", min_value=1, max_value=5, value=2, step=1)

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_audio_path = "temp_audio_file"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Transcribe"):
            with st.spinner("Processing..."):
                try:
                    # Load models
                    processor, model, lm_file = load_model_and_tokenizer()
                    decoder = get_decoder_ngram_model(processor.tokenizer, lm_file)

                    # Speaker diarization
                    segments = alternative_speaker_diarization(temp_audio_path, num_speakers=num_speakers)
                    
                    if not segments:
                        st.warning("No speech segments detected.")
                        return

                    # Process segments
                    final_transcriptions = process_segments(temp_audio_path, segments, model, processor, decoder)

                    # Display results
                    if final_transcriptions:
                        st.success("Transcription Completed!")
                        transcription_text = ""
                        for speaker, transcript in final_transcriptions:
                            st.markdown(f"**{speaker}:** {transcript}")
                            transcription_text += f"{speaker}: {transcript}\n"
                        
                        # Provide download link
                        st.download_button(
                            label="Download Transcription",
                            data=transcription_text,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No transcriptions available.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

            # Optionally, remove the temporary file after processing
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    else:
        st.info("Please upload an audio file to get started.")

if __name__ == '__main__':
    main()

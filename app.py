# -*- coding: utf-8 -*-
"""
Vietnamese end-to-end speech recognition using wav2vec 2.0
with long audio handling by chunking and resampling (Streamlit Version).
"""

import os
import zipfile
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from huggingface_hub import hf_hub_download
import numpy as np
import librosa  # for resampling
import streamlit as st


@st.cache_resource
def load_model_and_tokenizer(cache_dir='./cache/'):
    """
    Load the Wav2Vec2 processor, model, and download/extract the n-gram LM.
    Caching with st.cache_resource to avoid re-downloading.
    """
    st.write("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )

    st.write("Downloading language model...")
    lm_zip_file = hf_hub_download(
        repo_id="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        filename="vi_lm_4grams.bin.zip",
        cache_dir=cache_dir
    )

    st.write("Extracting language model...")
    with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    lm_file = os.path.join(cache_dir, 'vi_lm_4grams.bin')
    if not os.path.isfile(lm_file):
        raise FileNotFoundError(f"Language model file not found: {lm_file}")

    st.write("Processor, model, and language model loaded successfully.")
    return processor, model, lm_file


def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    """
    Build the beam search decoder with n-gram language model.
    """
    st.write("Building decoder with n-gram language model...")
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab_list = [token for _, token in sorted_vocab][:-2]

    st.write("Length of vocab_list before building alphabet:", len(vocab_list))
    alphabet = Alphabet.build_alphabet(vocab_list)
    
    lm_model = kenlm.Model(ngram_lm_path)
    
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    st.write("Decoder built successfully.")
    return decoder


def transcribe_chunk(model, processor, decoder, speech_chunk, sampling_rate):
    """
    Transcribe a single audio chunk.
    Ensures audio is mono, resampled to 16kHz, and in float32.
    """

    # Convert to mono if stereo
    if speech_chunk.ndim > 1:
        speech_chunk = np.mean(speech_chunk, axis=1)

    # Ensure float32
    speech_chunk = speech_chunk.astype(np.float32)

    # Resample if needed
    target_sr = 16000
    if sampling_rate != target_sr:
        speech_chunk = librosa.resample(speech_chunk, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr

    # Debug
    # st.write(f"Chunk shape after processing: {speech_chunk.shape}, SR: {sampling_rate}")

    # If the chunk is too small
    if len(speech_chunk) < 10:
        return ""

    input_values = processor(
        speech_chunk,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits[0]

    # Beam search decoding using the language model
    beam_search_output = decoder.decode(
        logits.cpu().detach().numpy(),
        beam_width=500
    )

    return beam_search_output


def transcribe_audio(processor, model, decoder, speech, sampling_rate):
    # Convert to mono if needed
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)
    speech = speech.astype(np.float32)

    # Define chunk length in seconds for long audio
    chunk_length_s = 30
    chunk_length_samples = int(chunk_length_s * sampling_rate)

    total_length = len(speech)
    num_chunks = int(np.ceil(total_length / chunk_length_samples))

    st.write(f"Splitting audio into {num_chunks} chunks of {chunk_length_s} seconds each (approx).")

    final_transcription = ""

    for i in range(num_chunks):
        start = i * chunk_length_samples
        end = min((i+1) * chunk_length_samples, total_length)
        speech_chunk = speech[start:end]

        st.write(f"Transcribing chunk {i+1}/{num_chunks}...")
        chunk_transcript = transcribe_chunk(model, processor, decoder, speech_chunk, sampling_rate)
        final_transcription += " " + chunk_transcript

    return final_transcription.strip()


def main():
    st.title("Vietnamese Speech-to-Text using Wav2Vec2 & n-gram LM")

    audio_file = st.file_uploader("Upload an audio file (WAV, MP3, etc.)", type=["wav", "mp3", "flac", "ogg"])

    if audio_file is not None:
        # Display player
        st.audio(audio_file)

        processor, model, lm_file = load_model_and_tokenizer('./cache/')
        decoder = get_decoder_ngram_model(processor.tokenizer, lm_file)

        if st.button("Transcribe"):
            # Try reading with soundfile
            try:
                speech, sampling_rate = sf.read(audio_file)
            except:
                # Fallback to librosa if soundfile fails
                speech, sampling_rate = librosa.load(audio_file, sr=None)

            st.write(f"Loaded audio: {len(speech)} samples at {sampling_rate} Hz")

            final_transcription = transcribe_audio(processor, model, decoder, speech, sampling_rate)
            st.subheader("Transcription:")
            st.write(final_transcription)


if __name__ == '__main__':
    main()

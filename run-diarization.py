# -*- coding: utf-8 -*-
"""
Vietnamese end-to-end speech recognition using wav2vec 2.0
with long audio handling by chunking, resampling, and speaker diarization.
"""

import os
import zipfile
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from huggingface_hub import hf_hub_download
import argparse
import numpy as np
import librosa  # for resampling
from pyannote.audio import Pipeline  # For speaker diarization

def load_model_and_tokenizer(cache_dir='./cache/'):
    """
    Load the Wav2Vec2 processor, model, and download/extract the n-gram LM.
    """
    print("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        cache_dir=cache_dir
    )

    print("Downloading language model...")
    lm_zip_file = hf_hub_download(
        repo_id="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        filename="vi_lm_4grams.bin.zip",
        cache_dir=cache_dir
    )

    print("Extracting language model...")
    with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    lm_file = os.path.join(cache_dir, 'vi_lm_4grams.bin')
    if not os.path.isfile(lm_file):
        raise FileNotFoundError(f"Language model file not found: {lm_file}")

    print("Processor, model, and language model loaded successfully.")
    return processor, model, lm_file

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    """
    Build the beam search decoder with n-gram language model.
    """
    print("Building decoder with n-gram language model...")
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab_list = [token for _, token in sorted_vocab][:-2]  # Exclude special tokens

    print("Length of vocab_list before building alphabet:", len(vocab_list))
    alphabet = Alphabet.build_alphabet(vocab_list)
    
    lm_model = kenlm.Model(ngram_lm_path)
    
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    print("Decoder built successfully.")
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

    # Just a sanity check print
    print(f"Chunk shape after processing: {speech_chunk.shape}, SR: {sampling_rate}")

    # If the chunk is empty or too small, skip
    if len(speech_chunk) < 10:
        # Just return empty transcript if too short
        print("Chunk too short to process, skipping...")
        return ""

    input_values = processor(
        speech_chunk,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_values

    # Another debug print
    print(f"Input values shape: {input_values.shape}")

    with torch.no_grad():
        logits = model(input_values).logits[0]

    # Beam search decoding using the language model
    beam_search_output = decoder.decode(
        logits.cpu().detach().numpy(),
        beam_width=500
    )

    return beam_search_output

def perform_diarization(audio_file, pipeline):
    """
    Perform speaker diarization on the audio file.
    Returns a list of segments with speaker labels.
    """
    print("Performing speaker diarization...")
    diarization = pipeline(audio_file)
    segments = []
    speaker_mapping = {}
    speaker_count = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_mapping:
            speaker_count += 1
            speaker_mapping[speaker] = f"Speaker {speaker_count}"
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker_mapping[speaker]
        })

    print("Diarization completed.")
    return segments

def main():
    parser = argparse.ArgumentParser(description="Vietnamese End-to-End Speech Recognition with Speaker Diarization")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()
    audio_file = args.audio

    print(f"Processing audio file: {audio_file}")
    if not os.path.isfile(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    if os.path.getsize(audio_file) == 0:
        print(f"Audio file is empty: {audio_file}")
        return

    cache_dir = './cache/'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model, tokenizer, and n-gram LM
    processor, model, lm_file = load_model_and_tokenizer(cache_dir)
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_file)

    # Initialize Speaker Diarization Pipeline
    print("Loading speaker diarization pipeline...")
    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=None)
    except Exception as e:
        print("Error loading diarization pipeline:", e)
        print("Ensure you have access to the pretrained model and necessary dependencies.")
        return

    # Perform diarization
    speaker_segments = perform_diarization(audio_file, diarization_pipeline)

    # Read full audio
    speech, sampling_rate = sf.read(audio_file)
    print(f"Loaded audio: {len(speech)} samples at {sampling_rate} Hz")

    # Convert to mono if needed
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)
    speech = speech.astype(np.float32)

    final_transcription = ""

    for idx, segment in enumerate(speaker_segments, start=1):
        start_sample = int(segment["start"] * sampling_rate)
        end_sample = int(segment["end"] * sampling_rate)
        speech_chunk = speech[start_sample:end_sample]

        print(f"Transcribing segment {idx}/{len(speaker_segments)}: {segment['speaker']} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
        chunk_transcript = transcribe_chunk(model, processor, ngram_lm_model, speech_chunk, sampling_rate)
        
        if chunk_transcript.strip() != "":
            final_transcription += f"{segment['speaker']}: {chunk_transcript.strip()}\n"

    # Print final result
    print("\nFinal transcription with speaker labels:")
    print(final_transcription.strip())

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Vietnamese end-to-end speech recognition using wav2vec 2.0
with automatic speaker diarization.
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
import librosa
from pyannote.audio import Pipeline

def load_model_and_tokenizer(cache_dir='./cache/'):
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

def get_decoder_ngram_model(_tokenizer, ngram_lm_path):
    print("Building decoder with n-gram language model...")
    vocab_dict = _tokenizer.get_vocab()
    sorted_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab_list = [token for _, token in sorted_vocab][:-2]  # Exclude special tokens

    alphabet = Alphabet.build_alphabet(vocab_list)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    print("Decoder built successfully.")
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
        # Optionally, you can skip padding and return an empty string
        # return ""

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

def automatic_speaker_diarization(audio_file):
    """
    Perform speaker diarization using pyannote.audio's pre-trained pipeline.
    Automatically detects the number of speakers.
    Returns a list of segments with speaker labels.
    """
    try:
        # Initialize the pre-trained speaker diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        
        # Apply the pipeline to the audio file
        diarization = pipeline(audio_file)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        return segments

    except Exception as e:
        print(f"Speaker diarization failed: {e}")
        # Fallback to a simple equal-length segmentation
        audio, sr = sf.read(audio_file)
        total_duration = len(audio) / sr
        # Estimate number of speakers using a heuristic or set a default
        # For simplicity, we'll set it to 2
        num_speakers = 2
        segment_duration = total_duration / num_speakers

        segments = []
        for i in range(num_speakers):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            segments.append((start, end, f"Speaker {i + 1}"))

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
            final_transcriptions.append((speaker_id, transcript))

    return final_transcriptions

def main():
    parser = argparse.ArgumentParser(description="Vietnamese End-to-End Speech Recognition with Automatic Speaker Diarization")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()
    audio_file = args.audio

    if not os.path.isfile(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    if os.path.getsize(audio_file) == 0:
        print(f"Audio file is empty: {audio_file}")
        return

    cache_dir = './cache/'
    os.makedirs(cache_dir, exist_ok=True)

    processor, model, lm_file = load_model_and_tokenizer(cache_dir)
    decoder = get_decoder_ngram_model(processor.tokenizer, lm_file)

    # Use the automatic diarization method
    segments = automatic_speaker_diarization(audio_file)
    final_transcriptions = process_segments(audio_file, segments, model, processor, decoder)

    # Map unique speaker labels to consistent numbering
    speaker_mapping = {}
    current_speaker = 1
    for speaker_id, _ in final_transcriptions:
        if speaker_id not in speaker_mapping:
            speaker_mapping[speaker_id] = f"Speaker {current_speaker}"
            current_speaker += 1

    for speaker_id, transcript in final_transcriptions:
        print(speaker_mapping[speaker_id] + ":", transcript)

if __name__ == '__main__':
    main()

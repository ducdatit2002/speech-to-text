# -*- coding: utf-8 -*-
"""
Vietnamese end-to-end speech recognition using wav2vec 2.0
with speaker diarization and timestamps (MM:SS format).
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
from pyAudioAnalysis import audioSegmentation as aS

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

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    print("Building decoder with n-gram language model...")
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab_list = [token for _, token in sorted_vocab][:-2]

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

    MIN_DURATION = 0.5  # seconds
    MIN_SAMPLES = int(MIN_DURATION * sampling_rate)

    if len(speech_chunk) < MIN_SAMPLES:
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
        y, sr = librosa.load(audio_file, sr=None)
        intervals = librosa.effects.split(y, top_db=30)  # Adjust top_db as needed

        # Merge very short intervals
        MIN_INTERVAL_DURATION = 0.5  # seconds
        MIN_SAMPLES = int(MIN_INTERVAL_DURATION * sr)
        merged_intervals = []
        for interval in intervals:
            if merged_intervals and (interval[0] - merged_intervals[-1][1]) < MIN_SAMPLES:
                merged_intervals[-1][1] = interval[1]
            else:
                merged_intervals.append([interval[0], interval[1]])

        segments = []
        for i, (start, end) in enumerate(merged_intervals):
            speaker_id = i % num_speakers
            start_time = start / sr
            end_time = end / sr
            segments.append((start_time, end_time, speaker_id))

        return segments

    except Exception as e:
        print(f"Speaker diarization failed: {e}")
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
        if transcript.strip():
            final_transcriptions.append((start, end, speaker_id, transcript))
    return final_transcriptions

def format_timestamp(seconds):
    # Định dạng thời gian thành MM:SS
    total_seconds = int(seconds)
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"

def main():
    parser = argparse.ArgumentParser(description="Vietnamese End-to-End Speech Recognition with Speaker Diarization")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file")
    parser.add_argument('--num_speakers', type=int, default=2, help="Number of speakers to detect")
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

    segments = alternative_speaker_diarization(audio_file, num_speakers=args.num_speakers)
    final_transcriptions = process_segments(audio_file, segments, model, processor, decoder)

    if not final_transcriptions:
        print("Không có bản phiên âm nào được tạo ra.")
        return

    merged_results = []
    # Khởi tạo với đoạn đầu tiên
    prev_start, prev_end, prev_speaker_id, prev_text = final_transcriptions[0]

    for i in range(1, len(final_transcriptions)):
        start, end, speaker_id, text = final_transcriptions[i]
        if speaker_id == prev_speaker_id:
            # Cùng speaker, gộp đoạn
            prev_end = end  # cập nhật thời gian kết thúc
            prev_text += " " + text
        else:
            # Khác speaker, in ra speaker trước và reset
            merged_results.append((prev_start, prev_end, prev_speaker_id, prev_text))
            prev_start, prev_end, prev_speaker_id, prev_text = start, end, speaker_id, text

    # Thêm đoạn cuối cùng
    merged_results.append((prev_start, prev_end, prev_speaker_id, prev_text))

    # Tạo full_transcript
    full_transcript = ""
    for start_time, end_time, speaker_id, transcript in merged_results:
        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)
        line = f"{start_str} - {end_str} - Speaker {speaker_id + 1}: {transcript}"
        full_transcript += line + "\n"
        print(line)

    # Lưu kết quả vào file txt có cùng tên với file input
    output_filename = audio_file + ".txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print(f"Đã lưu phiên âm vào file: {output_filename}")

if __name__ == '__main__':
    main()

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")

def transcribe_audio(audio_data):
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

import pyaudio
import numpy as np
import torch
import torch.cuda
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from queue import Queue
from threading import Thread
import time

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# Define available models
model_dict = {
    'en': 'Helsinki-NLP/opus-mt-ROMANCE-en',  # Various languages to English
    'es': 'Helsinki-NLP/opus-mt-en-es',  # English to Spanish
    'fr': 'Helsinki-NLP/opus-mt-en-fr',  # English to French
    'de': 'Helsinki-NLP/opus-mt-en-de',  # English to German
    'ja': 'Helsinki-NLP/opus-mt-ja-en',  # Japanese to English
}

def download_model(model_name):
    try:
        if not os.path.exists(os.path.join('~/.cache/huggingface/transformers', model_name.replace('/', '_'))):
            MarianMTModel.from_pretrained(model_name)
            MarianTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")

def load_translation_model(src_lang, tgt_lang):
    if tgt_lang == 'en':
        model_name = model_dict.get(src_lang, model_dict['en'])
    elif src_lang == 'en':
        model_name = model_dict.get(tgt_lang, model_dict['en'])
    else:
        model_name = model_dict['en']  # Default to translating to English
    download_model(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# Set up matplotlib for displaying captions
fig, ax = plt.subplots()
text_display = ax.text(0.5, 0.5, "", ha='center', va='center', fontsize=24, wrap=True)
ax.axis('off')

def update_caption(frame):
    text_display.set_text(frame)
    return [text_display]

def animate(queue):
    while True:
        yield queue.get()

def display_captions(queue):
    ani = animation.FuncAnimation(fig, update_caption, frames=animate(queue), interval=100, blit=True, cache_frame_data=False)
    plt.show()

def transcribe_audio(audio_data, src_lang):
    with torch.no_grad():
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = asr_model.generate(inputs, language=src_lang, task="transcribe")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def translate_text(text, src_lang, target_lang):
    if src_lang == target_lang:
        return text
    tokenizer, model = load_translation_model(src_lang, target_lang)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
        translated = model.generate(inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def process_audio_segment(audio_segment, queue, src_lang, target_lang):
    # Apply noise threshold
    if np.max(np.abs(audio_segment)) < 0.01:  # Adjust this threshold as needed
        return

    transcription = transcribe_audio(audio_segment, src_lang)
    if transcription.strip():  # Only process non-empty transcriptions
        print(f"Transcription: {transcription}")
        
        if src_lang != target_lang:
            translated_text = translate_text(transcription, src_lang, target_lang)
            print(f"Translated Text: {translated_text}")
            queue.put(f"{src_lang}: {transcription}\n{target_lang}: {translated_text}")
        else:
            queue.put(f"{src_lang}: {transcription}")

def capture_system_audio(queue, src_lang, target_lang):
    CHUNK = 48000  # Capture 3 seconds of audio at a time
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* Recording")
    
    audio_buffer = np.array([], dtype=np.float32)
    
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.float32)
        audio_buffer = np.concatenate((audio_buffer, audio_data))
        
        if len(audio_buffer) >= RATE * 5:  # Process 5 seconds of audio
            process_audio_segment(audio_buffer, queue, src_lang, target_lang)
            audio_buffer = np.array([], dtype=np.float32)  # Clear the buffer

def main():
    src_lang = input("Enter source language code (e.g., 'ja' for Japanese): ")
    target_lang = input("Enter target language code (e.g., 'en' for English): ")
    
    if src_lang not in model_dict and src_lang != 'en':
        print(f"Unsupported source language: {src_lang}. Defaulting to English.")
        src_lang = 'en'
    
    if target_lang not in model_dict and target_lang != 'en':
        print(f"Unsupported target language: {target_lang}. Defaulting to English.")
        target_lang = 'en'
    
    print("Starting audio capture and translation...")
    queue = Queue()
    
    audio_thread = Thread(target=capture_system_audio, args=(queue, src_lang, target_lang))
    audio_thread.daemon = True
    audio_thread.start()
    
    display_captions(queue)

if __name__ == "__main__":
    main()

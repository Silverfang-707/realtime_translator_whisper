import pyaudio
import numpy as np
import torch
import torch.cuda
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import langid
import os
from queue import Queue
from threading import Thread

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

# Set target language (can be changed by user)
target_language = 'en'  # Default target language

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

def transcribe_audio(audio_data):
    with torch.no_grad():
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = asr_model.generate(inputs, language="en", task="transcribe")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def detect_language(text):
    try:
        lang, confidence = langid.classify(text)
        if confidence > 0.5:
            return lang
        else:
            return 'unknown'
    except:
        return 'unknown'

def translate_text(text, src_lang, target_lang):
    if src_lang == target_lang or src_lang == 'unknown':
        return text
    tokenizer, model = load_translation_model(src_lang, target_lang)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
        translated = model.generate(inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def process_audio_segment(audio_segment, queue):
    # Apply noise threshold
    if np.max(np.abs(audio_segment)) < 0.01:  # Adjust this threshold as needed
        return

    transcription = transcribe_audio(audio_segment)
    if transcription.strip():  # Only process non-empty transcriptions
        print(f"Transcription: {transcription}")
        
        src_lang = detect_language(transcription)
        print(f"Detected Language: {src_lang}")
        
        if src_lang != target_language and src_lang != 'unknown':
            translated_text = translate_text(transcription, src_lang, target_language)
            print(f"Translated Text: {translated_text}")
            queue.put(f"{src_lang}: {transcription}\n{target_language}: {translated_text}")
        else:
            queue.put(f"{src_lang}: {transcription}")

def capture_system_audio(queue):
    CHUNK = 32000  # Capture 2 seconds of audio at a time
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
    
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.float32)
        process_audio_segment(audio_data, queue)

def main():
    global target_language
    target_language = input("Enter target language code (e.g., 'en' for English, 'es' for Spanish): ")
    if target_language not in model_dict:
        print(f"Unsupported language: {target_language}. Defaulting to English.")
        target_language = 'en'
    
    print("Starting audio capture and translation...")
    queue = Queue()
    
    audio_thread = Thread(target=capture_system_audio, args=(queue,))
    audio_thread.daemon = True
    audio_thread.start()
    
    display_captions(queue)

if __name__ == "__main__":
    main()
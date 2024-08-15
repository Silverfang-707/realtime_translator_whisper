from transformers import MarianMTModel, MarianTokenizer

# Load MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-es'  # Example: English to Spanish
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to("cuda")

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True).input_ids.to("cuda")
    translated = model.generate(inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

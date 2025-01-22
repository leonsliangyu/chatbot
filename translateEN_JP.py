
import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-jap")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

st.title(' English to Japanese Translation Bot ')
input_text = st.text_area('Enter your text to translate: ', key="input_text")

# Create a pipeline
encoded = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)


if input_text:
    output = model.generate(**encoded)
    st.subheader("Translation:")
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(translated_text)
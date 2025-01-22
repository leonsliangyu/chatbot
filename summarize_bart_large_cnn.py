

from transformers import AutoTokenizer, pipeline
import torch
import streamlit as st



# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

st.title(' Text Summarization Bot ')
input_text = st.text_area('Enter your text to summarize: ', key="input_text")

# Create a pipeline
pipe = pipeline("summarization", model=model_name, tokenizer=tokenizer)


# Check if input text exists and summarize
if input_text:
    max_input_length = 1024  # Maximum token limit for BART
    tokenized_length = len(pipe.tokenizer.encode(input_text))
    
    if tokenized_length > max_input_length:
        st.warning(f"Input text is too long! It has {tokenized_length} tokens. Please shorten it to {max_input_length} tokens or less.")
    else:
        try:
            # Generate the summary using the pipeline
            summary = pipe(input_text, max_length=130, min_length=30, do_sample=False)
            # Display the summary
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
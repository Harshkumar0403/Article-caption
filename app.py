import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the saved model and tokenizer
MODEL_PATH = "./t5_caption_model"

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# Streamlit App
st.title("News Caption Generator")
st.markdown("Provide a news article, and the app will generate the best caption for it!")

# Input text box for the news article
news_article = st.text_area("Enter the news article here:", height=200)

# Generate caption
if st.button("Generate Caption"):
    if news_article.strip() == "":
        st.error("Please enter a news article to generate a caption!")
    else:
        # Prepare the input for the model
        inputs = tokenizer.encode("summarize: " + news_article, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate output from the model
        outputs = model.generate(
            inputs,
            max_length=150,  # Maximum length of the caption
            num_beams=5,    # Beam search for better results
            early_stopping=True
        )
        
        # Decode the output and display the caption
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("Generated Caption:")
        st.write(caption)
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path to your saved model
MODEL_PATH = "./t5_caption_model"

# Load model and tokenizer (loaded once at startup)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_caption(news_article):
    if news_article.strip() == "":
        return "❌ Please enter a news article to generate a caption."

    # Prepare input
    inputs = tokenizer.encode(
        "summarize: " + news_article,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate caption
    outputs = model.generate(
        inputs,
        max_length=150,
        num_beams=5,
        early_stopping=True
    )

    # Decode output
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption


# Gradio UI
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter the news article here...",
        label="News Article"
    ),
    outputs=gr.Textbox(
        label="Generated Caption"
    ),
    title="📰 News Caption Generator",
    description="Provide a news article, and the model will generate the best caption for it!"
)

if __name__ == "__main__":
    demo.launch()

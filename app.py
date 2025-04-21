import streamlit as st
import torch
from transformers import pipeline, set_seed
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the animation (optional)
lottie_json = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jzqndnfe.json")

# Initialize model
joke_generator = pipeline("text-generation", model="gpt2", device=-1)
set_seed(42)

# Set Streamlit config
st.set_page_config(page_title="AI Joke Generator", page_icon="😂", layout="centered")

# Show header and animation
st.title("😂 AI Joke Generator")
st.markdown("Make your day better with a little AI-powered humor!")

if lottie_json:
    st_lottie(lottie_json, height=200)

# Joke categories
categories = [
    "Marriage", "Programming", "Animals", "Doctors", "Work",
    "School", "Relationship", "Technology", "Food", "Politics"
]

category = st.selectbox("🎯 Choose a joke topic:", categories)

# Session state to store jokes
if "joke_history" not in st.session_state:
    st.session_state.joke_history = []

# Generate joke
if st.button("😂 Generate Joke"):
    prompt = f"tell me a joke about {category}"
    with st.spinner("Generating joke..."):
        # Wrap the generation in a torch.no_grad() context to prevent unnecessary computation
        with torch.no_grad():
            output = joke_generator(prompt, max_length=100, do_sample=True, top_k=50)
            generated_text = output[0]['generated_text'].replace(prompt, "").strip()

    # Store joke in session
    st.session_state.joke_history.append(generated_text)

    # Display the joke
    st.success("Here’s your joke!")
    st.write(generated_text)
    st.code(generated_text, language='markdown')

    # Download option
    st.download_button(
        label="📥 Download Joke as .txt",
        data=generated_text,
        file_name="joke.txt",
        mime="text/plain"
    )

# Show history of generated jokes
if st.session_state.joke_history:
    with st.expander("📜 Joke History"):
        for idx, past_joke in enumerate(reversed(st.session_state.joke_history), 1):
            st.markdown(f"**{idx}.** {past_joke}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ by Muhammad Asif | Powered by 🤗 Transformers & Streamlit</p>",
    unsafe_allow_html=True
)

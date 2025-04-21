import streamlit as st
from transformers import pipeline,set_seed

# Load Hugging Face text generation pipeline
joke_generator = pipeline("text-generation", model="gpt2", framework="pt")
set_seed(42)

# Streamlit app setup
st.set_page_config(page_title="Joke Generator", page_icon="😂")
st.title("😂 AI Joke Generator (Hugging Face)")

# Predefined categories (optional — for user inspiration)
categories = [
    "Marriage", "Programming", "Animals", "Doctors", "Work",
    "School", "Relationship", "Technology", "Food", "Politics"
]

category = st.selectbox("Choose a joke topic:", categories)

if st.button("Generate Joke"):
    if not category:
        st.warning("Please select a joke category.")
    else:
        # Create prompt for the HF model
        prompt = f"tell me a joke about {category}"

        with st.spinner("Generating joke..."):
            output = joke_generator(prompt, max_length=64, do_sample=True, top_k=50)[0]['generated_text']
            

        st.success("Here’s your joke!")
        st.write(output)

        # Display joke as code block
        st.code(output, language='markdown')

        # Download option
        st.download_button(
            label="📥 Download Joke as .txt",
            data=output,
            file_name="joke.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ by Muhammad Asif | Powered by 🤗 Transformers & Streamlit</p>", unsafe_allow_html=True)

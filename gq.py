import streamlit as st
from groq import Groq
import base64
import os
import numpy as np

# --- UI: Title and uploader ---
st.title("🌿 Crop Identifier using Groq & LLaMA 4")
st.markdown("Upload an image and identify the crop using a multimodal model.")

# --- API key (optional field or use env var) ---
gorq_api_key = st.secrets.get("GROQ_API_KEY")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload an image of a crop:", type=["jpg", "jpeg", "png"])

# --- Optional prompt input ---
user_prompt = st.text_input("Ask something about the image (optional):", value="What is the fruit crop type in the image/")


model_options = {
    "LLaMA 4 Scout (default)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "LLaMA 4 maverick": "meta-llama/llama-4-maverick-17b-128e-instruct"
}
selected_model_name = st.selectbox("Choose LLaMA model:", list(model_options.keys()))
selected_model = model_options[selected_model_name]



# --- Process when button is clicked ---
if st.button("🔍 Identify Crop") and uploaded_file and gorq_api_key:
    # Convert image to base64
    base64_image = base64.b64encode(uploaded_file.read()).decode('utf-8')

    # Setup Groq client
    client = Groq(api_key=gorq_api_key)

    with st.spinner("Calling Groq LLaMA 4 model..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a crop identification expert. Based on the image provided, identify the crop and respond in JSON: {croptype: fruit}. if use ask to count the fruit you provide the court too"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=selected_model,
            )

            result = chat_completion.choices[0].message.content
            st.success("✅ Response Received!")
            st.code(result, language="json" if result.strip().startswith("{") else "text")

        except Exception as e:
            st.error(f"❌ Failed to get response from Groq API: {e}")

# --- Show image ---
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Optional note
st.caption("Powered by [Groq](https://console.groq.com) + Meta's LLaMA 4.")

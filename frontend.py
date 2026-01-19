import streamlit as st
import requests

# 1. Title and Setup
st.set_page_config(page_title="LLM Gateway", page_icon="🤖")
st.title("🤖 AI Chat Gateway")
st.write("This interface talks to your custom Qwen2.5 API on Render.")

# 2. Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    # Replace this default with your ACTUAL Render URL
    api_url = st.text_input("Backend URL", "https://llmops-gateway-latest.onrender.com/generate")
    st.info("Ensure the URL ends with /generate")

# 3. Chat Interface
# We store the chat history in "session_state" so it doesn't disappear
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Input Box
prompt = st.chat_input("What is on your mind?")

if prompt:
    # A. Display User Message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Call Your API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The payload must match your Pydantic model in main.py
                payload = {"prompt": prompt}
                
                # Send request to Render
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    # Extract the text
                    data = response.json()
                    bot_reply = data.get("response", "No response found.")
                    st.markdown(bot_reply)
                    
                    # Save context
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")

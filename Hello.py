import streamlit as st


st.set_page_config(page_title="Stable Diffusion", layout="wide")

st.title("Stable Diffusion Playground")

st.sidebar.info("Select a demo above.")

st.markdown("A Stable Diffusion WebUI based on [Diffusers](https://huggingface.co/docs/diffusers/index) and [Streamlit](https://streamlit.io/).")

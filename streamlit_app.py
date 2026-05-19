import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_code_diff import st_code_diff
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import time

@st.cache_resource
def load_model():
    model = MT5ForConditionalGeneration.from_pretrained('iliemihai/mt5-base-romanian-diacritics', cache_dir='cache/')
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained('iliemihai/mt5-base-romanian-diacritics', legacy=False, cache_dir='cache/')
    return tokenizer

def initialize_app():
    st.set_page_config(
        page_title="Diacritice",
        page_icon="public/favicon.ico",
        menu_items={
            "About": "### Diacritice",
        },
    )
    st.title("🖋️Dia-critic")

def generate_text(text):
    model = load_model()
    tokenizer = load_tokenizer()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

def main():
    initialize_app()
    input_text = st.text_area("Introduceți textul fără diacritice mai jos")
    st.write(f'{len(input_text)} caractere.')
    if st.button("Corectează diacritice"):
        if input_text != "":
            res = ''
            with st.spinner('Sarcină în desfășurare...'):
                # start task
                res = generate_text(input_text)
                with st.container(border=True):
                    st.markdown(res)
                    # diff_viewer(input_text, res, split_view=False)
                    result = st_code_diff(old_string=input_text, new_string=res, language="plaintext")
                    st.write(f"Changes detected: {result['isChanged']}")
        else:
            st.warning("Câmpul este gol!")

if __name__ == "__main__":
    main()

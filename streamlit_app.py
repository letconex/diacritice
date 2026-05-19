import streamlit as st
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
            "About": "### Adaugă diacritice",
        },
    )
    st.title("🖋️Diacritice")

def generate_text(text):
    model = load_model()
    tokenizer = load_tokenizer()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

def main():
    initialize_app()
    input_text = st.text_area("Introduceți textul fără diacritice!")
    st.write(f'{len(input_text)} caractere.')

    # --- BUTTON: only sets state ---
    if st.button("Corectează diacritice"):
        if input_text:
            with st.spinner("Sarcină în desfășurare..."):
                st.session_state["input"] = input_text
                st.session_state["res"] = generate_text(input_text)
        else:
            st.warning("Câmpul este gol!")
    
    # --- RENDER: always outside the button ---
    if "res" in st.session_state:
        with st.container(border=True):
            result = st_code_diff(
                old_string=st.session_state["input"],
                new_string=st.session_state["res"],
                language="plaintext",
                diff_style="char",
                output_format="side-by-side", # "line-by-line" "side-by-side"
                force_inline_comparison=True,
            )
            st.write(f"Modificări: {"Da" if result['isChanged'] else "Nu"} Adăugări: {result['addNum']} Ștergeri: {result['delNum']}")
            st.markdown(f""":violet-badge[Modificări: {'Da' if result['isChanged'] else 'Nu'}] :green-badge[Adăugări: {result['addNum']}] :red-badge[Ștergeri: {result['delNum']}]""")
            st.code(st.session_state["res"], language=None, wrap_lines=True) # to copy text
            # st.markdown(st.session_state["res"])
if __name__ == "__main__":
    main()

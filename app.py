# app.py
import streamlit as st
import torch
import os
import torch.nn.functional as F

# ---- import your model and tokenizer classes ----
from model_defs import Encoder, Decoder, Seq2Seq, greedy_decode_sentence, beam_search_decode_sentence
from tokenizer_bpe import CharTokenizer  # ✅ using your char-level tokenizer

# -----------------------------
# 🧠 Paths and setup
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "seq2seq_checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 🧩 Load Tokenizers
# -----------------------------
urdu_tok = CharTokenizer()
urdu_tok.load(os.path.join(MODELS_DIR, "char_urdu"))  # ✅ loads char_urdu_vocab.json

roman_tok = CharTokenizer()
roman_tok.load(os.path.join(MODELS_DIR, "char_roman"))  # ✅ loads char_roman_vocab.json

# -----------------------------
# 🧩 Load specific checkpoint (hardcoded)
# -----------------------------
def load_checkpoint(ckpt_name="best_model_epoch17_bleu94.50.pt"):  # ✅ hardcode your file name here
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_name}")
        st.stop()

    data = torch.load(ckpt_path, map_location=device)

    SRC_PAD = urdu_tok.vocab["<pad>"]
    TGT_PAD = roman_tok.vocab["<pad>"]
    EMB_DIM = 128
    HID_DIM = 256
    ENC_LAYERS = DEC_LAYERS = 2
    DROPOUT = 0.2

    enc = Encoder(input_dim=len(urdu_tok.vocab), emb_dim=EMB_DIM, hid_dim=HID_DIM,
                  n_layers=ENC_LAYERS, dropout=DROPOUT, pad_idx=SRC_PAD)
    dec = Decoder(output_dim=len(roman_tok.vocab), emb_dim=EMB_DIM,
                  enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM, n_layers=DEC_LAYERS,
                  dropout=DROPOUT, pad_idx=TGT_PAD)

    model = Seq2Seq(enc, dec, enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM, dec_n_layers=DEC_LAYERS).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


@st.cache_resource
def get_model():
    return load_checkpoint("best_model_epoch20_bleu95.24.pt")  # ✅ replace with your chosen file


model = get_model()

# -----------------------------
# 🗣️ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Urdu → Roman Urdu Translator", page_icon="🗣️", layout="centered")
st.title("🗣️ Urdu → Roman Urdu Translator")
st.markdown("Enter Urdu text below and get the Roman Urdu output instantly!")

user_input = st.text_area("✍️ Type Urdu sentence:", placeholder="مثلاً: محبت کبھی ختم نہیں ہوتی", height=100)

method = st.radio("Decoding Method:", ["Beam Search", "Greedy"], horizontal=True)

if st.button("Translate 🚀"):
    if user_input.strip():
        with st.spinner("Translating..."):
            if method == "Greedy":
                output = greedy_decode_sentence(user_input, model, urdu_tok, roman_tok)
            else:
                output = beam_search_decode_sentence(user_input, model, urdu_tok, roman_tok, beam_width=4)
        st.success("✅ Translation complete!")
        st.markdown(f"### Roman Urdu Output:\n> {output}")
    else:
        st.warning("Please enter Urdu text to translate.")

st.markdown("---")
st.caption("Built with ❤️ using PyTorch + Streamlit")

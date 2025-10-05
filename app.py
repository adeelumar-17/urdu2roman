# app.py
import streamlit as st
import torch
import os
import torch.nn.functional as F

# ---- import your model and tokenizer classes ----
from model_defs import Encoder, Decoder, Seq2Seq, greedy_decode_sentence, beam_search_decode_sentence
from tokenizer_bpe import BPETokenizer

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
urdu_tok = BPETokenizer(
    vocab_path=os.path.join(MODELS_DIR, "urdu_bpe_vocab.json"),
    merges_path=os.path.join(MODELS_DIR, "urdu_bpe_merges.txt")
)
roman_tok = BPETokenizer(
    vocab_path=os.path.join(MODELS_DIR, "roman_bpe_vocab.json"),
    merges_path=os.path.join(MODELS_DIR, "roman_bpe_merges.txt")
)

# -----------------------------
# 🧩 Load Specific Checkpoint
# -----------------------------
def load_checkpoint():
    # 💡 Replace this with your actual checkpoint filename
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_epoch17_bleu94.50.pt")

    if not os.path.exists(ckpt_path):
        st.error(f"❌ Checkpoint not found: {ckpt_path}")
        st.stop()

    data = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Model params (must match training)
    SRC_PAD = urdu_tok.pad_id
    TGT_PAD = roman_tok.pad_id
    EMB_DIM = 256
    HID_DIM = 512
    ENC_LAYERS = DEC_LAYERS = 2
    DROPOUT = 0.3

    enc = Encoder(input_dim=len(urdu_tok.vocab), emb_dim=EMB_DIM, hid_dim=HID_DIM,
                  n_layers=ENC_LAYERS, dropout=DROPOUT, pad_idx=SRC_PAD)
    dec = Decoder(output_dim=len(roman_tok.vocab), emb_dim=EMB_DIM,
                  enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM,
                  n_layers=DEC_LAYERS, dropout=DROPOUT, pad_idx=TGT_PAD)

    model = Seq2Seq(enc, dec, enc_hid_dim=HID_DIM,
                    dec_hid_dim=HID_DIM, dec_n_layers=DEC_LAYERS).to(device)
    model.load_state_dict(data["model_state"], strict=False)
    model.eval()
    return model

@st.cache_resource
def get_model():
    return load_checkpoint()

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

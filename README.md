# Urdu to Roman Urdu Transliteration

A PyTorch + Streamlit project for transliterating Urdu text into Roman Urdu using a character-level sequence-to-sequence model with Luong-style attention.

## What this repository contains

- Inference app built with Streamlit
- Character tokenizer used for both source (Urdu) and target (Roman Urdu)
- Encoder-decoder model definitions and decoding helpers
- Prebuilt model/tokenizer artifacts under `models/`
- Training workflow notebook (`urdu-ghazals.ipynb`)

## Project structure

```text
.
|-- app.py
|-- model_defs.py
|-- tokenizer_bpe.py
|-- requirements.txt
|-- urdu-ghazals.ipynb
`-- models/
    |-- char_urdu_vocab.json
    |-- char_roman_vocab.json
    |-- urdu_bpe_vocab.json
    |-- urdu_bpe_merges.json
    |-- roman_bpe_vocab.json
    |-- roman_bpe_merges.json
    `-- seq2seq_checkpoints/
        `-- best_model_epoch20_bleu95.24.pt
```

## How inference works

1. `app.py` loads char tokenizers from:
   - `models/char_urdu_vocab.json`
   - `models/char_roman_vocab.json`
2. It constructs a Seq2Seq model from `model_defs.py`.
3. It loads checkpoint `models/seq2seq_checkpoints/best_model_epoch20_bleu95.24.pt`.
4. The app translates user input using:
   - Greedy decoding, or
   - Beam Search option in UI (currently routed to the same greedy implementation in code).

## Installation

### 1) Create and activate a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

From the repository root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Core modules

- `app.py`
  - Streamlit interface
  - Model/tokenizer loading
  - Greedy/Beam decode switch
- `model_defs.py`
  - `Encoder` (BiLSTM)
  - `LuongAttention`
  - `Decoder` (LSTM + attention context)
  - `Seq2Seq` wrapper
  - `greedy_decode_sentence`, `beam_search_decode_sentence`
- `tokenizer_bpe.py`
  - `CharTokenizer` class
  - vocab build/save/load
  - character-level encode/decode

## Model details

The loaded inference model is configured in `app.py` as:

- Embedding size: 128
- Hidden size: 256
- Encoder layers: 2 (bidirectional LSTM)
- Decoder layers: 2 (LSTM)
- Dropout: 0.2

Checkpoint used by default:

- `best_model_epoch20_bleu95.24.pt`

## Training notebook notes

`urdu-ghazals.ipynb` contains the end-to-end experimentation/training pipeline, including:

- Building aligned Urdu/Roman pairs from source text files
- Cleaning/normalization
- Train/validation/test split
- Tokenizer creation and saving
- Seq2Seq training/evaluation and checkpointing

The notebook output indicates a dataset of about 21k pairs in one run.

## Known limitations

- Beam search in `model_defs.py` is currently a simplified stub that returns greedy output.
- Checkpoint path and filename are hardcoded in `app.py`.
- Repository includes BPE vocab/merge artifacts, but inference app currently uses character tokenizers.

## Troubleshooting

- If tokenizer load fails, verify `models/char_urdu_vocab.json` and `models/char_roman_vocab.json` exist.
- If checkpoint load fails, verify `models/seq2seq_checkpoints/best_model_epoch20_bleu95.24.pt` exists.
- If Streamlit is missing, reinstall dependencies:

```bash
pip install -r requirements.txt
```

## Future improvements

- Implement true beam search decoding
- Add CLI inference script (non-UI)
- Make checkpoint/tokenizer paths configurable via environment variables
- Add basic tests for tokenizer and decode functions

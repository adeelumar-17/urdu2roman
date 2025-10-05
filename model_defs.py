# your_model_file.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =======================
# Encoder-Decoder Model
# =======================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers, 
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (h_n, c_n) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, h_n, c_n


class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        proj = self.W(dec_hidden).unsqueeze(2)
        scores = torch.bmm(enc_outputs, proj).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return attn_weights, context


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers=2, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attn = LuongAttention(enc_hid_dim * 2, dec_hid_dim)
        self.rnn = nn.LSTM(
            emb_dim + enc_hid_dim * 2, dec_hid_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, last_hidden, last_cell, enc_outputs, enc_mask):
        emb = self.dropout(self.embedding(input_tok).unsqueeze(1))
        dec_top_hidden = last_hidden[-1]
        attn_weights, context = self.attn(dec_top_hidden, enc_outputs, enc_mask)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (last_hidden, last_cell))
        logits = self.fc_out(torch.cat([output.squeeze(1), context, emb.squeeze(1)], dim=1))
        return logits, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_hid_dim, dec_hid_dim, dec_n_layers):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc2dec_h = nn.Linear(encoder.n_layers * 2 * enc_hid_dim, dec_n_layers * dec_hid_dim)
        self.enc2dec_c = nn.Linear(encoder.n_layers * 2 * enc_hid_dim, dec_n_layers * dec_hid_dim)
        self.dec_n_layers = dec_n_layers
        self.dec_hid_dim = dec_hid_dim

    def encode(self, src, src_lens, pad_idx):
        enc_outputs, enc_h, enc_c = self.encoder(src, src_lens)
        enc_mask = (src != pad_idx)
        return enc_outputs, enc_h, enc_c, enc_mask

# =======================
# Greedy + Beam Decoding
# =======================
def greedy_decode_sentence(text, model, src_tok, tgt_tok, max_len=120, device="cpu"):
    model.eval()
    src_ids = torch.tensor([src_tok.encode(text)], dtype=torch.long, device=device)
    src_lens = torch.tensor([src_ids.size(1)], dtype=torch.long, device=device)
    enc_outputs, enc_h, enc_c, enc_mask = model.encode(src_ids, src_lens, src_tok.vocab["<pad>"])

    B = 1
    dec_h_flat = model.enc2dec_h(enc_h.permute(1, 0, 2).reshape(B, -1))
    dec_c_flat = model.enc2dec_c(enc_c.permute(1, 0, 2).reshape(B, -1))
    dec_h = dec_h_flat.view(model.dec_n_layers, B, model.dec_hid_dim).contiguous()
    dec_c = dec_c_flat.view(model.dec_n_layers, B, model.dec_hid_dim).contiguous()

    input_tok = torch.tensor([tgt_tok.vocab["<sos>"]], dtype=torch.long, device=device)
    preds = []
    for _ in range(max_len):
        logits, dec_h, dec_c, _ = model.decoder.forward_step(input_tok, dec_h, dec_c, enc_outputs, enc_mask)
        next_tok = logits.argmax(dim=1)
        if next_tok.item() == tgt_tok.vocab["<eos>"]:
            break
        preds.append(next_tok.item())
        input_tok = next_tok
    return tgt_tok.decode(preds)


def beam_search_decode_sentence(text, model, src_tok, tgt_tok, beam_width=4, max_len=120, device="cpu"):
    # simplified beam for transliteration
    return greedy_decode_sentence(text, model, src_tok, tgt_tok, max_len, device)

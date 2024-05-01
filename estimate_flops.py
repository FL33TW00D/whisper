from whisper.model import AudioEncoder, TextDecoder
import torch
from torch.utils.flop_counter import FlopCounterMode

"""
This script estimates the FLOPS for the Whisper model.

The encoder runs once per 30 second chunk.
The decoder can run a maximum of ~250 times per 30 second chunk.
"""

ENCODER_LARGE_V3 = {
    "n_mels": 128,
    "n_ctx": 1500,
    "n_state": 1280,
    "n_head": 20,
    "n_layer": 32,
}

DECODER_LARGE_V3 = {
    "n_vocab": 51866,
    "n_ctx": 448,
    "n_state": 1280,
    "n_head": 20,
    "n_layer": 32,
}

ENCODER_BASE = {
    "n_mels": 80,
    "n_ctx": 1500,
    "n_state": 512,
    "n_head": 8,
    "n_layer": 6,
}

ENCODER_CFG = {
    "base": ENCODER_BASE,
    "large_v3": ENCODER_LARGE_V3
}

def f(variant:str="base"):
    print("Estimating FLOPS for Whisper: ", variant)
    print("Encoder")
    encoder = AudioEncoder(**ENCODER_CFG[variant])
    decoder = TextDecoder(**DECODER_LARGE_V3)
    encoder_input = torch.randn(1, 128, 3000)
    encoder(encoder_input)

    print("Decoder")
    for _ in range(250):
        decoder_x = torch.randint(0, 51866, (1, 1))
        decoder_xa = torch.randn(1, 1500, 1280)
        decoder(decoder_x, decoder_xa)

flop_counter = FlopCounterMode(display=True)
with flop_counter:
    f("large_v3")

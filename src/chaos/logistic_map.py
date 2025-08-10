import numpy as np
from src.utils.metrics import calculate_metrics

def generate_chaotic_sequence(length, x0=0.7, r=3.99):
    """
    Generate a chaotic sequence using the logistic map.
    Returns a NumPy array of bits (0 or 1).
    """
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = r * x * (1 - x)
        seq[i] = 1 if x > 0.5 else 0
    return seq.astype(np.uint8)

def chaotic_encrypt(message_bits, x0=0.7, r=3.99):
    """
    XOR the message bits with a chaotic sequence to encrypt.
    """
    chaos_seq = generate_chaotic_sequence(len(message_bits), x0, r)
    encrypted_bits = np.bitwise_xor(message_bits, chaos_seq)
    return encrypted_bits

def chaotic_decrypt(encrypted_bits, x0=0.7, r=3.99):
    """
    XOR again with the same chaotic sequence to decrypt.
    """
    return chaotic_encrypt(encrypted_bits, x0, r)  # XOR is symmetric
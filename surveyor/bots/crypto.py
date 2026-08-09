"""Callback signature checks and AES payload handling for Feishu and WeCom.

Both platforms use AES-256-CBC but wrap it differently: Feishu prefixes the
ciphertext with the IV, WeCom derives the IV from the key and embeds a random
prefix, a length header and the receiver id inside the plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class CryptoError(ValueError):
    pass


def _unpad(data: bytes) -> bytes:
    if not data:
        raise CryptoError("empty plaintext")
    pad = data[-1]
    if pad < 1 or pad > 32 or pad > len(data):
        raise CryptoError(f"bad padding byte {pad}")
    return data[:-pad]


def _pad(data: bytes, block_size: int = 32) -> bytes:
    pad = block_size - (len(data) % block_size)
    pad = pad or block_size
    return data + bytes([pad]) * pad


def _aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def _aes_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()


# --------------------------------------------------------------------------
# Feishu / Lark
# --------------------------------------------------------------------------

def feishu_decrypt(encrypt_key: str, encrypted: str) -> str:
    """Decrypt the ``encrypt`` field of a Feishu event callback."""
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    try:
        blob = base64.b64decode(encrypted)
    except Exception as exc:
        raise CryptoError(f"payload is not base64: {exc}") from exc
    if len(blob) <= 16:
        raise CryptoError("payload too short")
    plaintext = _aes_cbc_decrypt(key, blob[:16], blob[16:])
    return _unpad(plaintext).decode("utf-8", errors="replace")


def feishu_signature(timestamp: str, nonce: str, encrypt_key: str, body: bytes) -> str:
    """Recompute the ``X-Lark-Signature`` header for an encrypted callback."""
    digest = hashlib.sha256()
    digest.update(timestamp.encode("utf-8"))
    digest.update(nonce.encode("utf-8"))
    digest.update(encrypt_key.encode("utf-8"))
    digest.update(body)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# WeCom (企业微信)
# --------------------------------------------------------------------------

def wecom_signature(token: str, timestamp: str, nonce: str, encrypted: str) -> str:
    """The ``msg_signature`` WeCom sends alongside every callback."""
    parts = sorted([token, timestamp, nonce, encrypted])
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


def _wecom_key(aes_key: str) -> bytes:
    # WeCom prints the 43-character key without its base64 padding.
    key = base64.b64decode(aes_key + "=")
    if len(key) != 32:
        raise CryptoError(f"EncodingAESKey must decode to 32 bytes, got {len(key)}")
    return key


def wecom_decrypt(aes_key: str, encrypted: str) -> tuple[str, str]:
    """Decrypt a WeCom payload. Returns ``(message, receive_id)``."""
    key = _wecom_key(aes_key)
    try:
        blob = base64.b64decode(encrypted)
    except Exception as exc:
        raise CryptoError(f"payload is not base64: {exc}") from exc

    plaintext = _unpad(_aes_cbc_decrypt(key, key[:16], blob))
    # Layout: 16 random bytes | 4-byte big-endian length | message | receive id
    if len(plaintext) < 20:
        raise CryptoError("plaintext too short")
    body = plaintext[16:]
    (length,) = struct.unpack(">I", body[:4])
    if length > len(body) - 4:
        raise CryptoError("declared length exceeds payload")
    message = body[4 : 4 + length].decode("utf-8", errors="replace")
    receive_id = body[4 + length :].decode("utf-8", errors="replace")
    return message, receive_id


def wecom_encrypt(aes_key: str, message: str, receive_id: str) -> str:
    """Build the ``Encrypt`` field for a passive WeCom reply."""
    key = _wecom_key(aes_key)
    payload = (
        os.urandom(16)
        + struct.pack(">I", len(message.encode("utf-8")))
        + message.encode("utf-8")
        + receive_id.encode("utf-8")
    )
    return base64.b64encode(_aes_cbc_encrypt(key, key[:16], _pad(payload))).decode()


def constant_time_equal(left: str, right: str) -> bool:
    return hashlib.sha256(left.encode()).digest() == hashlib.sha256(right.encode()).digest()

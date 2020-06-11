import filecmp
import os
import tempfile

from Crypto.Cipher import AES
from Crypto.Cipher._mode_eax import EaxMode

def encrypt_file(encryption_key: bytes, in_filepath: str, out_filepath: str) -> None:
    cipher = AES.new(encryption_key, AES.MODE_EAX)
    assert isinstance(cipher, EaxMode)
    with open(in_filepath, "rb") as f_in:
        ciphertext, tag = cipher.encrypt_and_digest(f_in.read())
    with open(out_filepath, "wb+") as f_out:
        assert len(cipher.nonce) == 16
        assert len(tag) == 16
        for data in (cipher.nonce, tag, ciphertext):
            f_out.write(data)
    if __debug__:
        with tempfile.TemporaryDirectory() as tempdir:
            decrypted_filename = os.path.join(tempdir, "decrypted_file.dat")
            decrypt_file(encryption_key, out_filepath, decrypted_filename)
            assert filecmp.cmp(in_filepath, decrypted_filename), "files do not match"

def decrypt_file(encryption_key: bytes, in_filepath: str, out_filepath: str) -> None:
    with open(in_filepath, "rb") as f_in:
        nonce, tag, ciphertext = (f_in.read(x) for x in (16, 16, -1))
    cipher = AES.new(encryption_key, AES.MODE_EAX, nonce)
    assert isinstance(cipher, EaxMode)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    with open(out_filepath, "wb+") as f_out:
        f_out.write(data)

def encrypt(encryption_key: bytes, in_bytes: bytes) -> bytes:
    cipher = AES.new(encryption_key, AES.MODE_EAX)
    assert isinstance(cipher, EaxMode)
    ciphertext, tag = cipher.encrypt_and_digest(in_bytes)
    return cipher.nonce + tag + ciphertext

def decrypt(encryption_key: bytes, encrypted_data: bytes) -> bytes:
    # use the derived key to decrypt the shared secret
    nonce, tag, ciphertext = encrypted_data[:16], encrypted_data[16:32], encrypted_data[32:]
    cipher = AES.new(encryption_key, AES.MODE_EAX, nonce)
    assert isinstance(cipher, EaxMode)
    return cipher.decrypt_and_verify(ciphertext, tag)

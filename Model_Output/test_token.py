#https://github.com/google/sentencepiece/blob/master/doc/experiments.md

import sentencepiece as spm
import os
import sys

# Example corpus
corpus = ["Hello, how are you?", "I am fine, thank you."]

# Write the corpus to a text file
corpus_file_path = "corpus.txt"
with open(corpus_file_path, "w") as f:
    for line in corpus:
        f.write(line + "\n")

# Check if the corpus file exists
if not os.path.exists(corpus_file_path):
    raise FileNotFoundError(f"The corpus file '{corpus_file_path}' does not exist.")

# Save original stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# Redirect stdout and stderr to /dev/null
with open(os.devnull, 'w') as devnull:
    sys.stdout = devnull
    sys.stderr = devnull

    # Train SentencePiece model with BPE
    spm.SentencePieceTrainer.Train(input=corpus_file_path, model_prefix='spm_model', vocab_size=26, model_type='bpe')

# Restore original stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

sp = spm.SentencePieceProcessor()
sp.Load(model_file='spm_model.model')

# Encode text
encoded_text = sp.Encode("Hello, how are you?", out_type= int)

# Decode text
 
decoded_text = sp.Decode(encoded_text, out_type= str)

print("Encoded text:", encoded_text)
print("Decoded text:", decoded_text)

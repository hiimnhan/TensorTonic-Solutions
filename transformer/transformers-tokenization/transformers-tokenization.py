import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        for i, t in enumerate(special_tokens):
            self.word_to_id[t] = i
            self.id_to_word[i] = t

        unique_tokens = set()

        for text in texts:
            text = text.lower().split()
            unique_tokens.update(text)

        for token in sorted(unique_tokens):
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        self.vocab_size = len(self.word_to_id)
                    
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        out = []

        for t in text.lower().split():
            unk_id = self.word_to_id.get(self.unk_token)
            out.append(self.word_to_id.get(t, unk_id))

        return out
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        out = []

        for id in ids:
            out.append(self.id_to_word.get(id, self.unk_token))

        return " ".join(out)

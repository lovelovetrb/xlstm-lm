from transformers import AutoTokenizer


def setup_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"})
    return tokenizer

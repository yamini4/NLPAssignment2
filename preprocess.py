# PART 3: NMT PREPROCESSING
# English -> Hindi
# Encoder-Decoder LSTM with Attention
import os
import re
import json
import unicodedata
import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/processed/parallel_corpus.csv"
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
MAX_SOURCE_LENGTH = 30
MAX_TARGET_LENGTH = 30
MIN_FREQUENCY = 3
RANDOM_SEED = 42
# ============================================================
# SPECIAL TOKENS
# ============================================================
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN
]
# ============================================================
# 1. LOAD DATASET
# ============================================================
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Please run data_preparation.py first."
        )
    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig"
    )
    print("=" * 70)
    print("DATASET LOADED")
    print("=" * 70)
    print(
        f"Number of sentence pairs: {len(df)}"
    )
    print(
        f"Columns: {list(df.columns)}"
    )
    return df
# ============================================================
# 2. UNICODE NORMALIZATION
# ============================================================
def normalize_unicode(text):
    text = str(text)
    return unicodedata.normalize(
        "NFC",
        text
    )
# ============================================================
# 3. TEXT CLEANING
# ============================================================
def clean_text(
    text,
    lowercase=False
):
    text = normalize_unicode(text)
    # Replace multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    # Remove leading/trailing spaces
    text = text.strip()
    # Lowercase English
    if lowercase:
        text = text.lower()
    return text
# ============================================================
# 4. ENGLISH TOKENIZATION
# ============================================================
def tokenize_english(text):
    """
    English tokenization.
    Example:
    "How are you?"
    ->
    ["how", "are", "you", "?"]
    "He changed his name."
    ->
    ["he", "changed", "his", "name", "."]
    Numbers are kept as separate tokens.
    Punctuation is kept as separate tokens.
    """
    text = normalize_unicode(text)
    text = text.lower()
    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE
    )
    return tokens
# ============================================================
# 5. HINDI TOKENIZATION
# ============================================================
def tokenize_hindi(text):
    """
    Hindi / Devanagari tokenization.
    Example:
    "यह एक घर है।"
    ->
    ["यह", "एक", "घर", "है", "।"]
    """
    text = normalize_unicode(text)
    tokens = re.findall(
        r"[\u0900-\u097F]+|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE
    )
    return tokens
# ============================================================
# 6. ADD SOS / EOS
# ============================================================
def add_special_tokens(tokens):
    return (
        [SOS_TOKEN]
        + tokens
        + [EOS_TOKEN]
    )
# ============================================================
# 7. PROCESS ENGLISH SENTENCES
# ============================================================
def process_source_sentences(df):
    processed = []
    for text in df["English"]:
        text = clean_text(
            text,
            lowercase=True
        )
        tokens = tokenize_english(
            text
        )
        tokens = add_special_tokens(
            tokens
        )
        processed.append(
            tokens
        )
    return processed
# ============================================================
# 8. PROCESS HINDI SENTENCES
# ============================================================
def process_target_sentences(df):
    processed = []
    for text in df["Hindi"]:
        text = clean_text(
            text,
            lowercase=False
        )
        tokens = tokenize_hindi(
            text
        )
        tokens = add_special_tokens(
            tokens
        )
        processed.append(
            tokens
        )
    return processed
# ============================================================
# 9. BUILD VOCABULARY
# ============================================================
def build_vocabulary(
    tokenized_sentences,
    min_frequency=MIN_FREQUENCY
):
    counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(
            sentence
        )
    vocabulary = {}
    # --------------------------------------------------------
    # Special tokens first
    # --------------------------------------------------------
    for token in SPECIAL_TOKENS:
        vocabulary[token] = len(
            vocabulary
        )
    # --------------------------------------------------------
    # Normal tokens
    # --------------------------------------------------------
    for token, frequency in counter.items():
        if frequency >= min_frequency:
            if token not in vocabulary:
                vocabulary[token] = len(
                    vocabulary
                )
    return vocabulary
# ============================================================
# 10. NUMERICALIZE
# ============================================================
def numericalize(
    tokens,
    vocabulary
):
    unk_index = vocabulary[
        UNK_TOKEN
    ]
    return [
        vocabulary.get(
            token,
            unk_index
        )
        for token in tokens
    ]
# ============================================================
# 11. PAD / TRUNCATE
# ============================================================
def pad_sequence(
    sequence,
    max_length,
    pad_index,
    eos_index
):
    sequence = list(sequence)
    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------
    if len(sequence) > max_length:
        sequence = sequence[
            :max_length
        ]
        # Ensure EOS is present
        sequence[-1] = eos_index
    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------
    while len(sequence) < max_length:
        sequence.append(
            pad_index
        )
    return sequence
# ============================================================
# 12. CONVERT SENTENCES TO TENSORS
# ============================================================
def create_tensor_dataset(
    tokenized_sentences,
    vocabulary,
    max_length
):
    pad_index = vocabulary[
        PAD_TOKEN
    ]
    eos_index = vocabulary[
        EOS_TOKEN
    ]
    numericalized_sentences = []
    for tokens in tokenized_sentences:
        # Convert tokens -> IDs
        ids = numericalize(
            tokens,
            vocabulary
        )
        # Pad / truncate
        ids = pad_sequence(
            ids,
            max_length,
            pad_index,
            eos_index
        )
        numericalized_sentences.append(
            ids
        )
    return torch.tensor(
        numericalized_sentences,
        dtype=torch.long
    )
# ============================================================
# 13. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
def split_dataset(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED
    )
    train_df = train_df.reset_index(
        drop=True
    )
    val_df = val_df.reset_index(
        drop=True
    )
    test_df = test_df.reset_index(
        drop=True
    )
    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)
    print(
        f"Training   : {len(train_df)}"
    )
    print(
        f"Validation : {len(val_df)}"
    )
    print(
        f"Testing    : {len(test_df)}"
    )
    return (
        train_df,
        val_df,
        test_df
    )
# ============================================================
# 14. SAVE CSV SPLITS
# ============================================================
def save_splits(
    train_df,
    val_df,
    test_df
):
    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )
    train_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "train.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
    val_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "validation.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
    test_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "test.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
    print("\nCSV splits saved.")
# ============================================================
# 15. SAVE VOCABULARY
# ============================================================
def save_vocabulary(
    vocabulary,
    filename
):
    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        filename
    )
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=2
        )
    print(
        f"Vocabulary saved: {path}"
    )
# ============================================================
# 16. SAVE TENSOR
# ============================================================
def save_tensor(
    tensor,
    filename
):
    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )
    path = os.path.join(
        ARTIFACTS_DIR,
        filename
    )
    torch.save(
        tensor,
        path
    )
    print(
        f"Tensor saved: {path}"
    )
# ============================================================
# 17. DISPLAY TOKENIZATION EXAMPLES
# ============================================================
def display_examples(
    train_df,
    source_tokens,
    target_tokens,
    source_tensor,
    target_tensor
):
    print("\n" + "=" * 70)
    print("TOKENIZATION AND NUMERICALIZATION EXAMPLES")
    print("=" * 70)
    count = min(
        5,
        len(train_df)
    )
    for i in range(count):
        print(
            f"\nExample {i + 1}"
        )
        print(
            "English:",
            train_df.iloc[i]["English"]
        )
        print(
            "English tokens:",
            source_tokens[i]
        )
        print(
            "English IDs:",
            source_tensor[i].tolist()
        )
        print(
            "Hindi:",
            train_df.iloc[i]["Hindi"]
        )
        print(
            "Hindi tokens:",
            target_tokens[i]
        )
        print(
            "Hindi IDs:",
            target_tensor[i].tolist()
        )
# ============================================================
# 18. TOKENIZATION SANITY CHECK
# ============================================================
def tokenization_sanity_check():
    print("\n" + "=" * 70)
    print("TOKENIZATION SANITY CHECK")
    print("=" * 70)
    examples = [
        "He changed his name in his writings because of the English people.",
        "Poornaprajna is awareness and understanding of the Whole.",
        "Public were enraged at the punitive measures taken by the administration."
    ]
    for text in examples:
        tokens = tokenize_english(text)
        print("\nEnglish:")
        print(text)
        print("Tokens:")
        print(tokens)
# ============================================================
# 19. MAIN
# ============================================================
def main():
    # --------------------------------------------------------
    # Step 1: Load dataset
    # --------------------------------------------------------
    df = load_dataset()
    # --------------------------------------------------------
    # Step 2: Split dataset
    # --------------------------------------------------------
    (
        train_df,
        val_df,
        test_df
    ) = split_dataset(df)
    # --------------------------------------------------------
    # Step 3: Save CSV splits
    # --------------------------------------------------------
    save_splits(
        train_df,
        val_df,
        test_df
    )
    # --------------------------------------------------------
    # Step 4: Tokenization sanity check
    # --------------------------------------------------------
    tokenization_sanity_check()
    # --------------------------------------------------------
    # Step 5: Tokenize training data
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOKENIZING TRAINING DATA")
    print("=" * 70)
    source_tokens = (
        process_source_sentences(
            train_df
        )
    )
    target_tokens = (
        process_target_sentences(
            train_df
        )
    )
    # --------------------------------------------------------
    # Step 6: Build vocabularies
    # --------------------------------------------------------
    vocabulary_source = (
        build_vocabulary(
            source_tokens
        )
    )
    vocabulary_target = (
        build_vocabulary(
            target_tokens
        )
    )
    print(
        f"\nSource vocabulary size: "
        f"{len(vocabulary_source)}"
    )
    print(
        f"Target vocabulary size: "
        f"{len(vocabulary_target)}"
    )
    # --------------------------------------------------------
    # Step 7: Save vocabularies
    # --------------------------------------------------------
    save_vocabulary(
        vocabulary_source,
        "source_vocab.json"
    )
    save_vocabulary(
        vocabulary_target,
        "target_vocab.json"
    )
    # --------------------------------------------------------
    # Step 8: Create training tensors
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("CREATING TRAINING TENSORS")
    print("=" * 70)
    train_source_tensor = (
        create_tensor_dataset(
            source_tokens,
            vocabulary_source,
            MAX_SOURCE_LENGTH
        )
    )
    train_target_tensor = (
        create_tensor_dataset(
            target_tokens,
            vocabulary_target,
            MAX_TARGET_LENGTH
        )
    )
    # --------------------------------------------------------
    # Step 9: Validation tensors
    # --------------------------------------------------------
    val_source_tokens = (
        process_source_sentences(
            val_df
        )
    )
    val_target_tokens = (
        process_target_sentences(
            val_df
        )
    )
    val_source_tensor = (
        create_tensor_dataset(
            val_source_tokens,
            vocabulary_source,
            MAX_SOURCE_LENGTH
        )
    )
    val_target_tensor = (
        create_tensor_dataset(
            val_target_tokens,
            vocabulary_target,
            MAX_TARGET_LENGTH
        )
    )
    # --------------------------------------------------------
    # Step 10: Test tensors
    # --------------------------------------------------------
    test_source_tokens = (
        process_source_sentences(
            test_df
        )
    )
    test_target_tokens = (
        process_target_sentences(
            test_df
        )
    )
    test_source_tensor = (
        create_tensor_dataset(
            test_source_tokens,
            vocabulary_source,
            MAX_SOURCE_LENGTH
        )
    )
    test_target_tensor = (
        create_tensor_dataset(
            test_target_tokens,
            vocabulary_target,
            MAX_TARGET_LENGTH
        )
    )
    # --------------------------------------------------------
    # Step 11: Save tensors
    # --------------------------------------------------------
    save_tensor(
        train_source_tensor,
        "train_source.pt"
    )
    save_tensor(
        train_target_tensor,
        "train_target.pt"
    )
    save_tensor(
        val_source_tensor,
        "val_source.pt"
    )
    save_tensor(
        val_target_tensor,
        "val_target.pt"
    )
    save_tensor(
        test_source_tensor,
        "test_source.pt"
    )
    save_tensor(
        test_target_tensor,
        "test_target.pt"
    )
    # --------------------------------------------------------
    # Step 12: Display examples
    # --------------------------------------------------------
    display_examples(
        train_df,
        source_tokens,
        target_tokens,
        train_source_tensor,
        train_target_tensor
    )
    # --------------------------------------------------------
    # Step 13: Print tensor shapes
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("TENSOR SHAPES")
    print("=" * 70)
    print(
        "Train source:",
        train_source_tensor.shape
    )
    print(
        "Train target:",
        train_target_tensor.shape
    )
    print(
        "Validation source:",
        val_source_tensor.shape
    )
    print(
        "Validation target:",
        val_target_tensor.shape
    )
    print(
        "Test source:",
        test_source_tensor.shape
    )
    print(
        "Test target:",
        test_target_tensor.shape
    )
    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3 PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nCreated files:")
    print(
        "data/processed/train.csv"
    )
    print(
        "data/processed/validation.csv"
    )
    print(
        "data/processed/test.csv"
    )
    print(
        "artifacts/source_vocab.json"
    )
    print(
        "artifacts/target_vocab.json"
    )
    print(
        "artifacts/train_source.pt"
    )
    print(
        "artifacts/train_target.pt"
    )
    print(
        "artifacts/val_source.pt"
    )
    print(
        "artifacts/val_target.pt"
    )
    print(
        "artifacts/test_source.pt"
    )
    print(
        "artifacts/test_target.pt"
    )
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
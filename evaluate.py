# PART 3: NMT EVALUATION
# English -> Hindi
# Encoder-Decoder LSTM with Attention

import os
import re
import json
import pandas as pd
import torch

from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction
)

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.csv"
)

SOURCE_VOCAB_FILE = os.path.join(
    ARTIFACTS_DIR,
    "source_vocab.json"
)

TARGET_VOCAB_FILE = os.path.join(
    ARTIFACTS_DIR,
    "target_vocab.json"
)

MODEL_FILE = os.path.join(
    ARTIFACTS_DIR,
    "best_model.pt"
)

RESULTS_FILE = os.path.join(
    ARTIFACTS_DIR,
    "evaluation_results.json"
)

TRANSLATIONS_FILE = os.path.join(
    ARTIFACTS_DIR,
    "sample_translations.txt"
)


# ============================================================
# MODEL CONFIGURATION
# MUST MATCH TRAINING
# ============================================================

MAX_SOURCE_LENGTH = 30
MAX_TARGET_LENGTH = 30

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.2


# ============================================================
# CPU EVALUATION
# Set to 0 to evaluate complete test set.
# ============================================================

NUM_EVALUATION_SAMPLES = 1000


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 1. LOAD VOCABULARY
# ============================================================

def load_vocabulary(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vocabulary not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    return vocabulary


# ============================================================
# 2. REVERSE VOCABULARY
# ============================================================

def create_reverse_vocabulary(vocabulary):

    return {
        int(index): token
        for token, index
        in vocabulary.items()
    }


# ============================================================
# 3. TOKENIZATION
# ============================================================

def tokenize_english(text):

    text = str(text).lower()

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


def tokenize_hindi(text):

    text = str(text)

    tokens = re.findall(
        r"[\u0900-\u097F]+|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


# ============================================================
# 4. NUMERICALIZE
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
# 5. PAD / TRUNCATE
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
# 6. PROCESS SOURCE SENTENCE
# ============================================================

def process_source(
    text,
    vocabulary
):

    tokens = tokenize_english(
        text
    )

    tokens = (
        [SOS_TOKEN]
        + tokens
        + [EOS_TOKEN]
    )

    ids = numericalize(
        tokens,
        vocabulary
    )

    ids = pad_sequence(
        ids,
        MAX_SOURCE_LENGTH,
        vocabulary[PAD_TOKEN],
        vocabulary[EOS_TOKEN]
    )

    return ids


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

def load_model(
    source_vocab,
    target_vocab
):

    print("\n" + "=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Create same architecture
    # --------------------------------------------------------

    model = create_model(
        source_vocab_size=len(
            source_vocab
        ),
        target_vocab_size=len(
            target_vocab
        ),
        device=DEVICE
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "\nTrained model loaded successfully."
    )

    return model


# ============================================================
# 8. TRANSLATE ONE SENTENCE
# ============================================================

def translate_sentence(
    model,
    sentence,
    source_vocab,
    target_vocab,
    target_index_to_token
):

    model.eval()

    # --------------------------------------------------------
    # Convert English -> IDs
    # --------------------------------------------------------

    source_ids = process_source(
        sentence,
        source_vocab
    )

    source_tensor = torch.tensor(
        source_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    with torch.inference_mode():

        encoder_outputs, hidden, cell = (
            model.encoder(
                source_tensor
            )
        )

    # --------------------------------------------------------
    # Start decoder with SOS
    # --------------------------------------------------------

    input_token = torch.tensor(
        [
            target_vocab[
                SOS_TOKEN
            ]
        ],
        dtype=torch.long,
        device=DEVICE
    )

    generated_tokens = []

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    for _ in range(
        MAX_TARGET_LENGTH - 1
    ):

        with torch.inference_mode():

            output, hidden, cell, _ = (
                model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )

        # ----------------------------------------------------
        # Greedy decoding
        # ----------------------------------------------------

        prediction = output.argmax(
            dim=1
        )

        prediction_id = prediction.item()

        # ----------------------------------------------------
        # EOS -> stop
        # ----------------------------------------------------

        if prediction_id == target_vocab[
            EOS_TOKEN
        ]:

            break

        # ----------------------------------------------------
        # Ignore PAD
        # ----------------------------------------------------

        if prediction_id != target_vocab[
            PAD_TOKEN
        ]:

            token = target_index_to_token.get(
                prediction_id,
                UNK_TOKEN
            )

            generated_tokens.append(
                token
            )

        input_token = prediction

    return generated_tokens


# ============================================================
# 9. DETOKENIZE
# ============================================================

def detokenize(tokens):

    text = " ".join(
        tokens
    )

    # Remove spaces before punctuation

    text = re.sub(
        r"\s+([।,!?;:])",
        r"\1",
        text
    )

    # Remove spaces around slash

    text = re.sub(
        r"\s*/\s*",
        "/",
        text
    )

    return text.strip()


# ============================================================
# 10. WORD ACCURACY
# ============================================================

def calculate_word_accuracy(
    predicted,
    reference
):

    if len(reference) == 0:

        return (
            1.0
            if len(predicted) == 0
            else 0.0
        )

    correct = 0

    for i in range(
        min(
            len(predicted),
            len(reference)
        )
    ):

        if predicted[i] == reference[i]:

            correct += 1

    return (
        correct /
        len(reference)
    )


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    test_df,
    source_vocab,
    target_vocab
):

    target_index_to_token = (
        create_reverse_vocabulary(
            target_vocab
        )
    )

    references = []
    hypotheses = []

    exact_matches = 0

    total_word_accuracy = 0.0

    total_generated_words = 0

    total_reference_words = 0

    total_unknown_tokens = 0

    # ========================================================
    # NEW: SOURCE UNK STATISTICS
    # ========================================================

    total_source_unknown_tokens = 0

    total_source_tokens = 0

    translation_records = []

    print("\n" + "=" * 70)
    print("EVALUATING TEST DATA")
    print("=" * 70)

    print(
        f"\nEvaluation samples: "
        f"{len(test_df)}"
    )

    # ========================================================
    # Evaluation loop
    # ========================================================

    for index, row in test_df.iterrows():

        source_text = str(
            row["English"]
        )

        reference_text = str(
            row["Hindi"]
        )

        # ====================================================
        # NEW: CHECK SOURCE ENGLISH <unk>
        # ====================================================

        source_tokens = tokenize_english(
            source_text
        )

        source_unknown_count = sum(
            1
            for token in source_tokens
            if source_vocab.get(
                token,
                source_vocab[UNK_TOKEN]
            )
            == source_vocab[UNK_TOKEN]
        )

        total_source_unknown_tokens += (
            source_unknown_count
        )

        total_source_tokens += (
            len(source_tokens)
        )

        # ====================================================
        # Translate
        # ====================================================

        predicted_tokens = (
            translate_sentence(
                model,
                source_text,
                source_vocab,
                target_vocab,
                target_index_to_token
            )
        )

        # ====================================================
        # Reference tokens
        # ====================================================

        reference_tokens = tokenize_hindi(
            reference_text
        )

        # ----------------------------------------------------
        # Remove special tokens
        # ----------------------------------------------------

        reference_tokens = [
            token
            for token in reference_tokens
            if token not in [
                PAD_TOKEN,
                SOS_TOKEN,
                EOS_TOKEN
            ]
        ]

        predicted_tokens = [
            token
            for token in predicted_tokens
            if token not in [
                PAD_TOKEN,
                SOS_TOKEN,
                EOS_TOKEN
            ]
        ]

        # ====================================================
        # Count TARGET <unk> tokens
        # ====================================================

        unknown_count = sum(
            1
            for token in predicted_tokens
            if token == UNK_TOKEN
        )

        total_unknown_tokens += (
            unknown_count
        )

        # ====================================================
        # Length statistics
        # ====================================================

        total_generated_words += (
            len(predicted_tokens)
        )

        total_reference_words += (
            len(reference_tokens)
        )

        # ====================================================
        # Text
        # ====================================================

        predicted_text = detokenize(
            predicted_tokens
        )

        reference_clean_text = detokenize(
            reference_tokens
        )

        # ====================================================
        # Exact match
        # ====================================================

        if (
            predicted_text.strip()
            ==
            reference_clean_text.strip()
        ):

            exact_matches += 1

        # ====================================================
        # Word accuracy
        # ====================================================

        word_accuracy = (
            calculate_word_accuracy(
                predicted_tokens,
                reference_tokens
            )
        )

        total_word_accuracy += (
            word_accuracy
        )

        # ====================================================
        # BLEU
        # ====================================================

        references.append(
            [reference_tokens]
        )

        hypotheses.append(
            predicted_tokens
        )

        # ====================================================
        # Save translation record
        # ====================================================

        translation_records.append(
            {
                "english":
                    source_text,

                "reference_hindi":
                    reference_clean_text,

                "predicted_hindi":
                    predicted_text,

                "word_accuracy":
                    word_accuracy
            }
        )

        # ====================================================
        # Progress
        # ====================================================

        if (
            (index + 1) % 100 == 0
        ):

            print(
                f"Evaluated "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

    # ========================================================
    # BLEU SCORE
    # ========================================================

    smoothing = (
        SmoothingFunction()
        .method4
    )

    bleu_score = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=smoothing
    )

    # ========================================================
    # METRICS
    # ========================================================

    total_samples = len(
        test_df
    )

    exact_match_accuracy = (
        exact_matches /
        total_samples
    )

    average_word_accuracy = (
        total_word_accuracy /
        total_samples
    )

    average_generated_length = (
        total_generated_words /
        total_samples
    )

    average_reference_length = (
        total_reference_words /
        total_samples
    )

    # ========================================================
    # TARGET <unk> PERCENTAGE
    # ========================================================

    if total_generated_words > 0:

        unk_percentage = (
            total_unknown_tokens /
            total_generated_words
        ) * 100

    else:

        unk_percentage = 0.0

    # ========================================================
    # NEW: SOURCE <unk> PERCENTAGE
    # ========================================================

    if total_source_tokens > 0:

        source_unk_percentage = (
            total_source_unknown_tokens /
            total_source_tokens
        ) * 100

    else:

        source_unk_percentage = 0.0

    return (
        bleu_score,
        exact_match_accuracy,
        average_word_accuracy,
        average_generated_length,
        average_reference_length,
        unk_percentage,
        source_unk_percentage,
        translation_records
    )


# ============================================================
# 12. SAVE TRANSLATION EXAMPLES
# ============================================================

def save_translation_examples(
    records,
    number_of_examples=20
):

    with open(
        TRANSLATIONS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            "ENGLISH -> HINDI TRANSLATION EXAMPLES\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        for i, record in enumerate(
            records[:number_of_examples],
            start=1
        ):

            file.write(
                f"\nExample {i}\n"
            )

            file.write(
                "-" * 70
                + "\n"
            )

            file.write(
                "English:\n"
            )

            file.write(
                record["english"]
                + "\n\n"
            )

            file.write(
                "Reference Hindi:\n"
            )

            file.write(
                record["reference_hindi"]
                + "\n\n"
            )

            file.write(
                "Predicted Hindi:\n"
            )

            file.write(
                record["predicted_hindi"]
                + "\n\n"
            )

            file.write(
                "Word Accuracy: "
                f"{record['word_accuracy']:.4f}\n"
            )

    print(
        f"\nSample translations saved at:"
        f"\n{TRANSLATIONS_FILE}"
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

def save_results(
    bleu_score,
    exact_match_accuracy,
    average_word_accuracy,
    average_generated_length,
    average_reference_length,
    unk_percentage,
    source_unk_percentage,
    total_samples
):

    results = {

        "test_samples":
            total_samples,

        "bleu_score":
            bleu_score,

        "bleu_score_percent":
            bleu_score * 100,

        "exact_match_accuracy":
            exact_match_accuracy,

        "exact_match_accuracy_percent":
            exact_match_accuracy * 100,

        "average_word_accuracy":
            average_word_accuracy,

        "average_word_accuracy_percent":
            average_word_accuracy * 100,

        "average_generated_length":
            average_generated_length,

        "average_reference_length":
            average_reference_length,

        "unknown_token_percentage":
            unk_percentage,

        "source_unknown_token_percentage":
            source_unk_percentage
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nEvaluation results saved at:"
        f"\n{RESULTS_FILE}"
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ENGLISH -> HINDI NMT EVALUATION")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # ========================================================
    # Artifacts directory
    # ========================================================

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # Load vocabularies
    # ========================================================

    print(
        "\nLoading vocabularies..."
    )

    source_vocab = load_vocabulary(
        SOURCE_VOCAB_FILE
    )

    target_vocab = load_vocabulary(
        TARGET_VOCAB_FILE
    )

    print(
        f"Source vocabulary: "
        f"{len(source_vocab)}"
    )

    print(
        f"Target vocabulary: "
        f"{len(target_vocab)}"
    )

    # ========================================================
    # Print special token IDs
    # ========================================================

    print(
        "\nSpecial tokens:"
    )

    print(
        f"PAD: "
        f"{target_vocab[PAD_TOKEN]}"
    )

    print(
        f"UNK: "
        f"{target_vocab[UNK_TOKEN]}"
    )

    print(
        f"SOS: "
        f"{target_vocab[SOS_TOKEN]}"
    )

    print(
        f"EOS: "
        f"{target_vocab[EOS_TOKEN]}"
    )

    # ========================================================
    # Load test data
    # ========================================================

    print(
        "\nLoading test data..."
    )

    test_df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Test samples available: "
        f"{len(test_df)}"
    )

    # ========================================================
    # Limit evaluation
    # ========================================================

    if (
        NUM_EVALUATION_SAMPLES
        and
        len(test_df)
        > NUM_EVALUATION_SAMPLES
    ):

        # Use FIRST N samples.
        # This makes evaluation deterministic.

        test_df = test_df.iloc[
            :NUM_EVALUATION_SAMPLES
        ].reset_index(
            drop=True
        )

    print(
        f"Test samples evaluated: "
        f"{len(test_df)}"
    )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(
        source_vocab,
        target_vocab
    )

    # ========================================================
    # Evaluate
    # ========================================================

    (
        bleu_score,
        exact_match_accuracy,
        average_word_accuracy,
        average_generated_length,
        average_reference_length,
        unk_percentage,
        source_unk_percentage,
        translation_records
    ) = evaluate_model(
        model,
        test_df,
        source_vocab,
        target_vocab
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("NMT EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nBLEU Score:"
        f" {bleu_score:.4f}"
    )

    print(
        f"BLEU Score (%):"
        f" {bleu_score * 100:.2f}%"
    )

    print(
        f"\nExact Match Accuracy:"
        f" {exact_match_accuracy:.4f}"
    )

    print(
        f"Exact Match Accuracy (%):"
        f" {exact_match_accuracy * 100:.2f}%"
    )

    print(
        f"\nAverage Word Accuracy:"
        f" {average_word_accuracy:.4f}"
    )

    print(
        f"Average Word Accuracy (%):"
        f" {average_word_accuracy * 100:.2f}%"
    )

    print(
        f"\nAverage Generated Length:"
        f" {average_generated_length:.2f}"
    )

    print(
        f"Average Reference Length:"
        f" {average_reference_length:.2f}"
    )

    print(
        f"\nTarget <unk> Percentage:"
        f" {unk_percentage:.2f}%"
    )

    # ========================================================
    # NEW: DISPLAY SOURCE UNK
    # ========================================================

    print(
        f"\nSource <unk> Percentage:"
        f" {source_unk_percentage:.2f}%"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        bleu_score,
        exact_match_accuracy,
        average_word_accuracy,
        average_generated_length,
        average_reference_length,
        unk_percentage,
        source_unk_percentage,
        len(test_df)
    )

    # ========================================================
    # SAVE TRANSLATIONS
    # ========================================================

    save_translation_examples(
        translation_records,
        number_of_examples=20
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nCreated files:"
    )

    print(
        "artifacts/evaluation_results.json"
    )

    print(
        "artifacts/sample_translations.txt"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
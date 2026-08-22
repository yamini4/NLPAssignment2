# ============================================================
# NMT FLASK APPLICATION
# English -> Hindi
# Encoder-Decoder LSTM with Attention
# ============================================================

from flask import Flask, render_template, request
import os
import re
import json
import torch

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS_DIR = "artifacts"

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

MAX_SOURCE_LENGTH = 30
MAX_TARGET_LENGTH = 30

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
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD VOCABULARY
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

        return json.load(file)


source_vocab = load_vocabulary(
    SOURCE_VOCAB_FILE
)

target_vocab = load_vocabulary(
    TARGET_VOCAB_FILE
)


# ============================================================
# REVERSE TARGET VOCABULARY
# ============================================================

target_index_to_token = {
    int(index): token
    for token, index
    in target_vocab.items()
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING NMT MODEL")
print("=" * 70)

print(
    f"Source vocabulary size: "
    f"{len(source_vocab)}"
)

print(
    f"Target vocabulary size: "
    f"{len(target_vocab)}"
)

print(
    f"Device: {DEVICE}"
)


model = create_model(
    source_vocab_size=len(source_vocab),
    target_vocab_size=len(target_vocab),
    device=DEVICE
)


if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Trained model not found: {MODEL_FILE}"
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("\nTrained model loaded successfully.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_english(text):

    text = str(text).lower()

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


# ============================================================
# NUMERICALIZATION
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
# PAD / TRUNCATE
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

        # Make sure EOS is present
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
# PROCESS SOURCE SENTENCE
# ============================================================

def process_source(text):

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
        source_vocab
    )

    ids = pad_sequence(
        ids,
        MAX_SOURCE_LENGTH,
        source_vocab[PAD_TOKEN],
        source_vocab[EOS_TOKEN]
    )

    return ids


# ============================================================
# DETOKENIZE HINDI
# ============================================================

def detokenize(tokens):

    text = " ".join(tokens)

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
# TRANSLATION FUNCTION
# ============================================================

def translate_sentence(sentence):

    if not sentence.strip():

        return ""


    # --------------------------------------------------------
    # English -> IDs
    # --------------------------------------------------------

    source_ids = process_source(
        sentence
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
    # Decoder starts with SOS
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
    # Generate Hindi tokens
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
        # Stop at EOS
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


    # --------------------------------------------------------
    # Convert tokens -> sentence
    # --------------------------------------------------------

    translation = detokenize(
        generated_tokens
    )


    return translation


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    english_text = ""
    hindi_translation = ""

    if request.method == "POST":

        english_text = request.form.get(
            "english_text",
            ""
        ).strip()

        if english_text:

            try:

                hindi_translation = (
                    translate_sentence(
                        english_text
                    )
                )

            except Exception as error:

                hindi_translation = (
                    f"Translation error: {error}"
                )


    return render_template(
        "index.html",
        english_text=english_text,
        hindi_translation=hindi_translation
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model": "English -> Hindi NMT",
        "device": str(DEVICE)
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ENGLISH -> HINDI NMT FLASK APPLICATION")
    print("=" * 70)

    print(
        "\nOpen in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
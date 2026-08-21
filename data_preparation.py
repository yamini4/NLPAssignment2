
# PART 2: DATASET DOWNLOAD, CLEANING AND PREPARATION
# English -> Hindi NMT

import os
import re
import unicodedata
import pandas as pd
from datasets import load_dataset

# CONFIGURATION

# Verified English-Hindi parallel corpus
DATASET_NAME = "cfilt/iitb-english-hindi"
# Number of sentence pairs required for the project
MAX_SAMPLES = 50000
# Reproducibility
RANDOM_SEED = 42
# Output directory
OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)
# Final CSV file
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "parallel_corpus.csv"
)

# 1. LOAD DATASET FROM HUGGING FACE

def load_parallel_dataset():
    print("=" * 70)
    print("LOADING ENGLISH-HINDI DATASET")
    print("=" * 70)
    print(
        f"\nDataset: {DATASET_NAME}"
    )
    print(
        "\nDownloading/loading training split..."
    )
    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )
    print(
        "\nDataset loaded successfully."
    )
    print(
        f"Total available sentence pairs: "
        f"{len(dataset)}"
    )
    print(
        f"Dataset columns: "
        f"{dataset.column_names}"
    )
    return dataset

# 2. SELECT ONLY REQUIRED NUMBER OF SENTENCE PAIRS

def sample_dataset(dataset):
    print("\n" + "=" * 70)
    print("SELECTING DATASET SUBSET")
    print("=" * 70)
    total_rows = len(dataset)
    print(
        f"Total available rows: "
        f"{total_rows}"
    )
    if total_rows > MAX_SAMPLES:
        print(
            f"\nSelecting {MAX_SAMPLES} "
            f"sentence pairs..."
        )
        # Shuffle the dataset
        dataset = dataset.shuffle(
            seed=RANDOM_SEED
        )
        # Select first 50,000 rows
        dataset = dataset.select(
            range(MAX_SAMPLES)
        )
    else:
        print(
            "\nDataset is smaller than the "
            f"requested {MAX_SAMPLES} rows."
        )
    print(
        f"\nSelected rows: "
        f"{len(dataset)}"
    )
    return dataset

# 3. EXTRACT ENGLISH AND HINDI FROM TRANSLATION FIELD

def convert_to_dataframe(dataset):
    print("\n" + "=" * 70)
    print("CONVERTING DATASET TO DATAFRAME")
    print("=" * 70)
    print(
        "\nThe dataset contains a "
        "'translation' field."
    )
    print(
        "Extracting English (en) and Hindi (hi)..."
    )
    # Convert selected 50,000 rows to Pandas
    raw_df = dataset.to_pandas()
    print(
        f"\nRaw DataFrame shape: "
        f"{raw_df.shape}"
    )
    print(
        f"Raw columns: "
        f"{list(raw_df.columns)}"
    )
    # --------------------------------------------------------
    # Check translation column
    # --------------------------------------------------------
    if "translation" not in raw_df.columns:
        raise ValueError(
            "Expected 'translation' column "
            "was not found.\n"
            f"Available columns: "
            f"{list(raw_df.columns)}"
        )
    # --------------------------------------------------------
    # Extract English and Hindi
    # --------------------------------------------------------
    english_sentences = []
    hindi_sentences = []
    for translation in raw_df["translation"]:
        if not isinstance(
            translation,
            dict
        ):
            continue
        english = translation.get(
            "en",
            ""
        )
        hindi = translation.get(
            "hi",
            ""
        )
        english_sentences.append(
            english
        )
        hindi_sentences.append(
            hindi
        )
    # Create final DataFrame
    df = pd.DataFrame(
        {
            "English": english_sentences,
            "Hindi": hindi_sentences
        }
    )
    print(
        f"\nExtracted DataFrame shape: "
        f"{df.shape}"
    )
    print(
        "\nColumns:"
    )
    print(
        list(df.columns)
    )
    return df

# 4. UNICODE NORMALIZATION

def normalize_unicode(text):
    if not isinstance(
        text,
        str
    ):
        text = str(text)
    # NFC normalization is important
    # for Hindi Devanagari text
    text = unicodedata.normalize(
        "NFC",
        text
    )
    return text

# 5. TEXT CLEANING

def clean_text(text):
    # Unicode normalization
    text = normalize_unicode(
        text
    )
    # Replace multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    # Remove leading/trailing spaces
    text = text.strip()
    return text

# 6. CLEAN DATASET

def clean_dataset(df):
    print("\n" + "=" * 70)
    print("CLEANING DATASET")
    print("=" * 70)
    initial_rows = len(df)
    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------
    df = df.dropna(
        subset=[
            "English",
            "Hindi"
        ]
    )
    print(
        "Removed missing rows:",
        initial_rows - len(df)
    )
    # --------------------------------------------------------
    # Convert columns to strings
    # --------------------------------------------------------
    df["English"] = (
        df["English"]
        .astype(str)
    )
    df["Hindi"] = (
        df["Hindi"]
        .astype(str)
    )
    # --------------------------------------------------------
    # Unicode normalization + cleaning
    # --------------------------------------------------------
    print(
        "\nApplying Unicode normalization..."
    )
    df["English"] = (
        df["English"]
        .apply(clean_text)
    )
    df["Hindi"] = (
        df["Hindi"]
        .apply(clean_text)
    )
    # --------------------------------------------------------
    # Remove empty sentences
    # --------------------------------------------------------
    before_empty = len(df)
    df = df[
        (df["English"].str.len() > 0)
        &
        (df["Hindi"].str.len() > 0)
    ]
    print(
        "Removed empty rows:",
        before_empty - len(df)
    )
    # --------------------------------------------------------
    # Remove duplicate sentence pairs
    # --------------------------------------------------------
    before_duplicates = len(df)
    df = df.drop_duplicates(
        subset=[
            "English",
            "Hindi"
        ]
    )
    print(
        "Removed duplicate rows:",
        before_duplicates - len(df)
    )
    # --------------------------------------------------------
    # Remove extremely long sentences
    # --------------------------------------------------------
    #
    # Assignment recommends short-to-medium
    # sentences around 5-30 tokens.
    #
    # We use a reasonable maximum of 50 words
    # to avoid removing useful sentences too aggressively.
    # --------------------------------------------------------
    before_length_filter = len(df)
    df["English_Words"] = (
        df["English"]
        .str.split()
        .str.len()
    )
    df["Hindi_Words"] = (
        df["Hindi"]
        .str.split()
        .str.len()
    )
    df = df[
        (df["English_Words"] <= 50)
        &
        (df["Hindi_Words"] <= 50)
    ]
    # Remove temporary columns
    df = df.drop(
        columns=[
            "English_Words",
            "Hindi_Words"
        ]
    )
    print(
        "Removed overly long rows:",
        before_length_filter - len(df)
    )
    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------
    df = df.reset_index(
        drop=True
    )
    print(
        "\nFinal cleaned rows:",
        len(df)
    )
    return df

# 7. DISPLAY SAMPLE SENTENCES

def display_samples(
    df,
    number_of_samples=10
):
    print("\n" + "=" * 70)
    print("SAMPLE ENGLISH-HINDI SENTENCE PAIRS")
    print("=" * 70)
    sample_count = min(
        number_of_samples,
        len(df)
    )
    for i in range(
        sample_count
    ):
        print(
            f"\nExample {i + 1}"
        )
        print(
            "English:",
            df.iloc[i]["English"]
        )
        print(
            "Hindi  :",
            df.iloc[i]["Hindi"]
        )

# 8. SAVE DATASET

def save_dataset(df):
    print("\n" + "=" * 70)
    print("SAVING DATASET")
    print("=" * 70)
    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    # Save CSV
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )
    # --------------------------------------------------------
    # Verify CSV was created
    # --------------------------------------------------------
    if os.path.exists(
        OUTPUT_FILE
    ):
        file_size = os.path.getsize(
            OUTPUT_FILE
        )
        print(
            "\nCSV CREATED SUCCESSFULLY!"
        )
        print(
            "\nFile location:"
        )
        print(
            os.path.abspath(
                OUTPUT_FILE
            )
        )
        print(
            f"\nNumber of sentence pairs: "
            f"{len(df)}"
        )
        print(
            f"File size: "
            f"{file_size / (1024 * 1024):.2f} MB"
        )
    else:
        raise RuntimeError(
            "\nERROR: CSV file was not created."
        )

# 9. VERIFY SAVED CSV

def verify_saved_csv():
    print("\n" + "=" * 70)
    print("VERIFYING SAVED CSV")
    print("=" * 70)
    if not os.path.exists(
        OUTPUT_FILE
    ):
        raise FileNotFoundError(
            f"CSV not found at:\n"
            f"{os.path.abspath(OUTPUT_FILE)}"
        )
    # Read first few rows
    test_df = pd.read_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        nrows=5
    )
    print(
        "\nCSV verification successful."
    )
    print(
        "\nCSV columns:"
    )
    print(
        list(test_df.columns)
    )
    print(
        "\nFirst 5 sentence pairs:"
    )
    print(
        test_df.to_string(
            index=False
        )
    )

# 10. MAIN PIPELINE

def main():
    # --------------------------------------------------------
    # STEP 1
    # Load dataset
    # --------------------------------------------------------
    dataset = (
        load_parallel_dataset()
    )
    # --------------------------------------------------------
    # STEP 2
    # Select 50,000 rows
    # --------------------------------------------------------
    dataset = (
        sample_dataset(
            dataset
        )
    )
    # --------------------------------------------------------
    # STEP 3
    # Extract English-Hindi
    # --------------------------------------------------------
    df = (
        convert_to_dataframe(
            dataset
        )
    )
    # --------------------------------------------------------
    # STEP 4
    # Clean dataset
    # --------------------------------------------------------
    df = (
        clean_dataset(
            df
        )
    )
    # --------------------------------------------------------
    # STEP 5
    # Display examples
    # --------------------------------------------------------
    display_samples(
        df
    )
    # --------------------------------------------------------
    # STEP 6
    # Save CSV
    # --------------------------------------------------------
    save_dataset(
        df
    )
    # --------------------------------------------------------
    # STEP 7
    # Verify CSV
    # --------------------------------------------------------
    verify_saved_csv()
    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(
        "\nYour processed dataset is ready:"
    )
    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

# RUN PROGRAM

if __name__ == "__main__":
    main()
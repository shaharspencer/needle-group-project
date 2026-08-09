import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_text_files_recursively(folder_path, file_extensions=(".txt", ".md")):
    """
    Recursively scans a directory for files matching the given extensions
    and returns a list of relative file paths.
    """
    base_dir = Path(folder_path)
    file_paths = []

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory '{folder_path}' does not exist.")

    for file_path in base_dir.rglob("*"):
        if file_path.is_file():
            if not file_extensions or file_path.suffix.lower() in file_extensions:
                file_paths.append(str(file_path))

    return sorted(file_paths)

def compute_folder_tfidf(folder_path, output_csv="tfidf_results.csv"):
    file_paths = load_text_files_recursively(folder_path, file_extensions=(".txt", ".md"))

    if not file_paths:
        print("[!] No matching text files found in the specified directory.")
        return None

    print(f"[+] Found {len(file_paths)} files. Calculating TF-IDF...")

    vectorizer = TfidfVectorizer(
        input="filename",
        encoding="utf-8",
        decode_error="ignore",
        stop_words="english",
        max_features=500
    )

    tfidf_matrix = vectorizer.fit_transform(file_paths)
    row_labels = [os.path.relpath(p, folder_path) for p in file_paths]

    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=row_labels
    )

    df_tfidf.round(4).to_csv(output_csv)
    print(f"[✔] Saved TF-IDF matrix to '{output_csv}'")

    return df_tfidf

def print_top_k_words_per_episode(df_tfidf, top_n=5):
    """
    Extracts and prints the top N words with the highest TF-IDF scores for each row (episode).
    """
    print(f"\n================ TOP {top_n} WORDS PER EPISODE ================")
    
    for episode, row in df_tfidf.iterrows():
        # Retrieve the N highest scoring words for this row
        top_words = row.nlargest(top_n)
        
        # Format word and score pairs
        formatted_terms = ", ".join([f"{word} ({score:.4f})" for word, score in top_words.items()])
        
        print(f"\n📄 Episode: {episode}")
        print(f"   Top Words: {formatted_terms}")

if __name__ == "__main__":
    TARGET_FOLDER = r"C:\Users\bard1\Documents\year5\needle in data haystack\project\datasets\transcripts\Game of Thrones"

    df = compute_folder_tfidf(TARGET_FOLDER)
    if df is not None:
        print_top_k_words_per_episode(df, top_n=5)
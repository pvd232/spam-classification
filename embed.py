import re

# Uncomment to download for first run
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
from math import log

s_dir, h_dir = os.path.join(os.path.curdir, "enron1/spam"), os.path.join(
    os.path.curdir, "enron1/ham"
)
s_len, h_len = len(os.listdir(s_dir)), len(os.listdir(h_dir))

tr_cov = 0.8  # train / test split
s_stop, h_stop = round(s_len * tr_cov), round(h_len * tr_cov)


def pre_process(filepath: str) -> list[str]:
    # Try utf‑8, fall back to latin‑1
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()

    # Remove punctuation, lowercase
    clean = re.sub(r"[^\w\s]", "", text).lower()

    # Tokenize the cleaned text
    tokens: list[str] = wordpunct_tokenize(clean)
    return tokens


def lemmanize(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    clean_tokens = [lemmatizer.lemmatize(token, "n") for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(token, "v") for token in clean_tokens]

    # Remove numbers
    clean_tokens = [re.sub(r"\b[0-9]+\b", "<NUM>", token) for token in clean_tokens]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    clean_tokens = [token for token in clean_tokens if token not in stop_words]
    return clean_tokens


def build_vocab(vocab: dict[str, int], corpus: list[str]) -> dict[str, int]:
    for token in corpus:
        if token not in vocab.keys():
            vocab[token] = len(vocab)


def build_index2token(
    index2token: dict[int, str], vocab: dict[str, int]
) -> dict[int, str]:
    for token in vocab.keys():
        index2token[vocab[token]] = token


def build_bow_vector(
    corp: list[list[str]], idx2token: dict[int, str], vocab: dict[str, int], test: bool
) -> list[int]:
    sequence = []
    for sample in corp:
        for token in sample:
            idx = vocab.get(token)
            sequence.append(idx)

    vector = [0] * len(idx2token)
    for token_idx in sequence:
        if token_idx not in idx2token:
            if not test:
                raise ValueError("Wrong sequence index found!")
            else:
                continue
        else:
            vector[token_idx] += 1
    return vector


def tok(
    filepath: str, vocab: dict[str, int], i_vocab: dict[int, str], test: bool
) -> list[str]:
    corpus_tokens: list[list[str]] = []
    tokens = pre_process(filepath)
    cleaned_sequence = lemmanize(tokens=tokens)
    corpus_tokens.append(cleaned_sequence)
    if not test:
        build_vocab(vocab=vocab, corpus=cleaned_sequence)
        build_index2token(vocab=vocab, index2token=i_vocab)
    return corpus_tokens


def tokenize(
    dir: str, vocab: dict[str, int], i_vocab: dict[int, str], stop: int
) -> list[str]:
    files = os.listdir(dir)
    corpus_tokens: list[list[str]] = []
    for i in range(stop):
        filename = files[i]
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            corpus_tokens += tok(
                filepath=filepath, vocab=vocab, i_vocab=i_vocab, test=False
            )
    return corpus_tokens


def classify(msg_vec: list[int], params) -> int:
    scores = {}
    for cls in ("spam", "ham"):
        logp = log(params[cls]["prior"])
        θ = params[cls]["theta"]
        for i, count in enumerate(msg_vec):
            if count > 0:
                logp += count * log(θ[i])
        scores[cls] = logp

    # pick the best
    return int(max(scores, key=scores.get) == "spam")


def naiive_bayes(
    spam_counts: list[int],
    ham_counts: list[int],
    V: int,
    vocab: dict[str, int],
    i_vocab: dict[int, str],
):
    spam_total, ham_total = 0, 0
    for c in spam_counts:
        spam_total += c
    for c in ham_counts:
        ham_total += c

    params = {"spam": {"prior": 0.3, "theta": ""}, "ham": {"prior": 0.7, "theta": ""}}

    alpha = 1
    total_tokens_spam = sum(spam_counts)
    denom_spam = total_tokens_spam + alpha * V
    theta_spam = [(spam_counts[i] + alpha) / denom_spam for i in range(V)]

    total_tokens_ham = sum(ham_counts)
    denom_ham = total_tokens_ham + alpha * V
    theta_ham = [(ham_counts[i] + alpha) / denom_ham for i in range(V)]

    params["spam"]["theta"] = theta_spam
    params["ham"]["theta"] = theta_ham

    spam_acc = test(
        start=s_stop + 1,
        dir=s_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="spam",
    )
    ham_acc = test(
        start=h_stop + 1,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
    )
    print(f"spam acc: {round(spam_acc*100,2)}%")
    print(f"ham acc: {round(ham_acc*100,2)}%")


def ro(x1: list[int], x2: list[int]):
    pass


def k_NN():

    pass


def test(
    start: int,
    dir: str,
    vocab: dict[str, int],
    i_vocab: dict[int, str],
    params: dict,
    cls: str,
) -> float:
    files = os.listdir(dir)
    n = len(files)

    right, wrong = 0, 0
    for i in range(start + 1, n):
        filename = files[i]
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            sample_tokens = tok(
                filepath=filepath, vocab=vocab, i_vocab=i_vocab, test=True
            )
            embedding = build_bow_vector(
                corp=sample_tokens, idx2token=i_vocab, vocab=vocab, test=True
            )

            dec = classify(msg_vec=embedding, params=params)
            if cls == "spam":
                if dec == 1:
                    right += 1
                else:
                    wrong += 1
            else:
                if dec == 0:
                    right += 1
                else:
                    wrong += 1

    acc = right / (right + wrong)
    return acc


def main():
    vocab, i_vocab = {}, {}
    spam_corp: list[list[str]] = tokenize(
        dir=s_dir, vocab=vocab, i_vocab=i_vocab, stop=s_stop
    )
    ham_corp: list[list[str]] = tokenize(
        dir=h_dir, vocab=vocab, i_vocab=i_vocab, stop=h_stop
    )

    spam_counts = build_bow_vector(
        corp=spam_corp, idx2token=i_vocab, vocab=vocab, test=False
    )
    ham_counts = build_bow_vector(
        corp=ham_corp, idx2token=i_vocab, vocab=vocab, test=False
    )

    naiive_bayes(
        spam_counts=spam_counts,
        ham_counts=ham_counts,
        V=len(vocab),
        vocab=vocab,
        i_vocab=i_vocab,
    )


main()

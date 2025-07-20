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
import statistics
import numpy as np

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
    corp: list[list[str]],
    idx2token: dict[int, str],
    vocab: dict[str, int],
    test: bool,
    flat: bool,
    median: bool,
) -> list[int] | list[list[int]]:
    V = len(idx2token)

    if flat:
        # collapse all emails into one big sequence
        flat_idxs = [
            vocab[tok]
            for sample in corp
            for tok in sample
            if vocab.get(tok) in idx2token
        ]
        vec = [0] * V
        for idx in flat_idxs:
            vec[idx] += 1
        return vec

    else:
        # build one BOW vector per email
        per_doc = []
        for sample in corp:
            vec = [0] * V
            for tok in sample:
                idx = vocab.get(tok)
                if not test and idx is None:
                    raise ValueError(f"Unknown token {tok}")
                if idx is not None and idx < V:
                    vec[idx] += 1
            per_doc.append(vec)
        if median:
            return [statistics.median(doc[j] for doc in per_doc) for j in range(V)]
        else:
            return per_doc


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


def classify(msg_vec: list[int], params: dict, model: str) -> int:
    if model == "n_bayes":
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
    elif model == "nn":
        # turn your test vector into a 1‑D NumPy array
        x = np.array(msg_vec)  # shape (V,)
        best_dists = {}
        for cls in ("spam", "ham"):
            M = params[cls]  # shape (N_cls, V)
            p = params["p"]
            if p == 1:
                # cityblock (L1)
                dists = np.abs(M - x).sum(axis=1)
            elif p == 2:
                # Euclidean
                dists = np.linalg.norm(M - x, ord=2, axis=1)
            else:
                # L-infinity
                dists = np.abs(M - x).max(axis=1)

            best_dists[cls] = dists.min()

        # pick whichever class has the smaller distance
        return int(min(best_dists, key=best_dists.get) == "spam")
    else:
        node: TreeNode = params["root"]
        prev = None
        while node:
            prev = node
            curr = msg_vec[node.feature]
            if curr <= node.thresh:
                node = node.left
            else:
                node = node.right
        return prev.label


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
        model="n_bayes",
    )
    ham_acc = test(
        start=h_stop + 1,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
        model="n_bayes",
    )
    print(f"spam acc: {round(spam_acc*100,2)}%")
    print(f"ham acc: {round(ham_acc*100,2)}%")


def k_NN(
    vocab: dict[str, int],
    i_vocab: dict[int, str],
    model: str,
    p: int,
    spam_count: list[int] = None,
    ham_count: list[int] = None,
):
    ham_acc, spam_acc = None, None
    params = {"spam": spam_count, "ham": ham_count, "p": p}
    spam_acc = test(
        start=s_stop + 1,
        dir=s_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="spam",
        model=model,
    )
    ham_acc = test(
        start=h_stop + 1,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
        model=model,
    )
    print(f"spam acc: {round(spam_acc*100,2)}%")
    print(f"ham acc: {round(ham_acc*100,2)}%")


def median_tree(vocab: dict[str, int], i_vocab: dict[int, str], model: str, root: dict):
    params = {"root": root}
    spam_acc = test(
        start=s_stop + 1,
        dir=s_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="spam",
        model=model,
    )
    ham_acc = test(
        start=h_stop + 1,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
        model=model,
    )
    print(f"spam acc: {round(spam_acc*100,2)}%")
    print(f"ham acc: {round(ham_acc*100,2)}%")


def test(
    start: int,
    dir: str,
    vocab: dict[str, int],
    i_vocab: dict[int, str],
    params: dict,
    cls: str,
    model: str,
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
                corp=sample_tokens,
                idx2token=i_vocab,
                vocab=vocab,
                test=True,
                flat=True,
                median=False,
            )

            dec = classify(msg_vec=embedding, params=params, model=model)
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


class TreeNode:
    def __init__(self, feature=None, thresh=None, left=None, right=None, label=None):
        self.feature = feature  # index j
        self.thresh = thresh  # split threshold
        self.left = left  # TreeNode or None
        self.right = right  # TreeNode or None
        self.label = label  # only for leaves


def fit_median_tree(corp, labels, vocab, i_vocab, max_depth=10, min_size=5, depth=0):
    # If pure or too small, make a leaf:
    if len(set(labels)) == 1 or len(labels) <= min_size or depth >= max_depth:
        # majority vote
        label = max(set(labels), key=labels.count)
        return TreeNode(label=label)

    # 1) turn each doc into a BOW vector
    vecs = build_bow_vector(corp, i_vocab, vocab, test=True, flat=False, median=False)

    # 2) pick feature j (here cycle by depth)
    j = depth % len(i_vocab)

    # 3) get median thresholds for this subset
    thresholds = build_bow_vector(
        corp, i_vocab, vocab, test=True, flat=False, median=True
    )
    t_j = thresholds[j]

    # 4) split the data
    left_idx = [i for i, v in enumerate(vecs) if v[j] <= t_j]
    right_idx = [i for i, v in enumerate(vecs) if v[j] > t_j]

    # if one side empty, force leaf
    if not left_idx or not right_idx:
        label = max(set(labels), key=labels.count)
        return TreeNode(label=label)

    left_corp = [corp[i] for i in left_idx]
    left_lbls = [labels[i] for i in left_idx]
    right_corp = [corp[i] for i in right_idx]
    right_lbls = [labels[i] for i in right_idx]

    # 5) recurse
    left_node = fit_median_tree(
        left_corp, left_lbls, vocab, i_vocab, max_depth, min_size, depth + 1
    )
    right_node = fit_median_tree(
        right_corp, right_lbls, vocab, i_vocab, max_depth, min_size, depth + 1
    )

    return TreeNode(feature=j, thresh=t_j, left=left_node, right=right_node)


def main():
    vocab, i_vocab = {}, {}
    spam_corp: list[list[str]] = tokenize(
        dir=s_dir, vocab=vocab, i_vocab=i_vocab, stop=s_stop
    )
    ham_corp: list[list[str]] = tokenize(
        dir=h_dir, vocab=vocab, i_vocab=i_vocab, stop=h_stop
    )

    spam_counts_b = build_bow_vector(
        corp=spam_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        model="n_bayes",
    )
    ham_counts_b = build_bow_vector(
        corp=ham_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        model="n_bayes",
    )

    naiive_bayes(
        spam_counts=spam_counts_b,
        ham_counts=ham_counts_b,
        V=len(vocab),
        vocab=vocab,
        i_vocab=i_vocab,
    )

    s_cnt_knn_b: list[list[int]] = build_bow_vector(
        corp=spam_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=False,
        median=False,
    )
    h_cnt_knn_b: list[list[int]] = build_bow_vector(
        corp=ham_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=False,
        median=False,
    )
    train_spam_mat = np.array(s_cnt_knn_b)  # shape (S, V)
    train_ham_mat = np.array(h_cnt_knn_b)  # shape (H, V)
    k_NN(
        vocab=vocab,
        i_vocab=i_vocab,
        spam_count=train_spam_mat,
        ham_count=train_ham_mat,
        model="nn",
        p=1,
    )

    labels = [1] * len(spam_corp) + [0] * len(ham_corp)
    all_corp = spam_corp + ham_corp
    root = fit_median_tree(corp=all_corp, labels=labels, vocab=vocab, i_vocab=i_vocab)

    median_tree(
        vocab=vocab,
        i_vocab=i_vocab,
        model="tree",
        root=root,
    )


main()

import re

# Uncomment to download for first run
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import time
from math import log
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import Self

s_dir, h_dir = os.path.join(os.curdir, "enron1/spam"), os.path.join(
    os.curdir, "enron1/ham"
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
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_index2token(
    index2token: dict[int, str], vocab: dict[str, int]
) -> dict[int, str]:
    for token, idx in vocab.items():
        index2token[idx] = token
    return index2token


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
        flat_idxs = [vocab[tok] for sample in corp for tok in sample if tok in vocab]
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
) -> list[list[str]]:
    tokens = pre_process(filepath)
    cleaned_sequence = lemmanize(tokens=tokens)
    if not test:
        build_vocab(vocab=vocab, corpus=cleaned_sequence)
        build_index2token(index2token=i_vocab, vocab=vocab)
    return [cleaned_sequence]


def tokenize(
    dir: str, vocab: dict[str, int], i_vocab: dict[int, str], stop: int
) -> list[list[str]]:
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
        return int(max(scores, key=scores.get) == "spam")

    elif model == "nn":
        x = np.array(msg_vec)
        best = {}
        for cls in ("spam", "ham"):
            M = params[cls]
            p = params["p"]
            if p == 1:
                d = np.abs(M - x).sum(axis=1)
            elif p == 2:
                d = np.linalg.norm(M - x, ord=2, axis=1)
            else:
                d = np.abs(M - x).max(axis=1)
            best[cls] = d.min()
        return int(min(best, key=best.get) == "spam")

    elif model == "kd_nn":
        root = params["root"]
        label, _ = kd_nearest(root, np.array(msg_vec), p=params["p"])
        return label

    elif model == "tree":  # decision‐tree
        node = params["root"]
        while node.label is None:
            if msg_vec[node.feature] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.label


def naiive_bayes(
    spam_counts: list[int],
    ham_counts: list[int],
    V: int,
    vocab: dict[str, int],
    i_vocab: dict[int, str],
):
    params = {
        "spam": {"prior": s_stop / (s_len + h_len), "theta": None},
        "ham": {"prior": h_stop / (s_len + h_len), "theta": None},
    }
    alpha = 1

    denom_spam = sum(spam_counts) + alpha * V
    denom_ham = sum(ham_counts) + alpha * V

    params["spam"]["theta"] = [(spam_counts[i] + alpha) / denom_spam for i in range(V)]
    params["ham"]["theta"] = [(ham_counts[i] + alpha) / denom_ham for i in range(V)]

    spam_acc = test(
        start=s_stop,
        dir=s_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="spam",
        model="n_bayes",
    )
    ham_acc = test(
        start=h_stop,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
        model="n_bayes",
    )
    print(f"spam acc: {spam_acc*100:.2f}%")
    print(f"ham acc:  {ham_acc*100:.2f}%")


class KDNode:

    def __init__(
        self,
        point: np.ndarray,
        label: int,
        axis: int,
        left: Self | None,
        right: Self | None,
    ):
        self.point = point  # 1D numpy array
        self.label = label  # its class (e.g. 0 or 1)
        self.axis = axis  # which coordinate we split on
        self.left = left  # KDNode or None
        self.right = right  # KDNode or None


def build_kdtree(
    points: np.ndarray, labels: list[int], depth: int = 0
) -> KDNode | None:
    """Recursive median split KD-tree builder."""
    if len(points) == 0:
        return None

    d = points.shape[1]
    axis = depth % d

    # sort point indices by the chosen axis
    idxs = points[:, axis].argsort()
    median = len(idxs) // 2
    m = idxs[median]

    # build node
    return KDNode(
        point=points[m],
        label=labels[m],
        axis=axis,
        left=build_kdtree(
            points[idxs[:median]], [labels[i] for i in idxs[:median]], depth + 1
        ),
        right=build_kdtree(
            points[idxs[median + 1 :]],
            [labels[i] for i in idxs[median + 1 :]],
            depth + 1,
        ),
    )


def kd_nearest(
    node: KDNode, x: np.ndarray, best: tuple[int, float] | None = None, p: float = 2.0
) -> tuple[int, float]:
    """
    Recursively search for the nearest neighbor of x under L_p,
    returning (best_label, best_dist).
    """

    if node is None:
        return best

    # 1) compute dist between x and this node’s point (a scalar)
    diff_vec = node.point - x
    if p == 1:
        dist = np.abs(diff_vec).sum()
    elif p == 2:
        dist = np.linalg.norm(diff_vec, ord=2)
    elif p == np.inf:
        dist = np.abs(diff_vec).max()
    else:
        # in case someone passes 3 or something
        dist = np.linalg.norm(diff_vec, ord=p)

    # 2) update best if this node is closer
    if best is None or dist < best[1]:
        best = (node.label, dist)

    # 3) choose which side of the plane to visit first
    axis = node.axis
    diff_along_axis = x[axis] - node.point[axis]
    if diff_along_axis < 0:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    # 4) recurse into the “closer” side
    best = kd_nearest(first, x, best, p)

    # 5) check whether we need to visit the other side
    #    (i.e. hypersphere of radius best[1] crosses the splitting plane)
    if abs(diff_along_axis) <= best[1]:
        best = kd_nearest(second, x, best, p)

    return best


def k_NN(
    vocab: dict[str, int],
    i_vocab: dict[int, str],
    model: str,
    p: int,
    spam_count: np.ndarray,
    ham_count: np.ndarray,
    root: KDNode = None,
):
    if root:
        params = {"root": root, "p": p}
    else:
        params = {"spam": spam_count, "ham": ham_count, "p": p}
    spam_acc = test(
        start=s_stop,
        dir=s_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="spam",
        model=model,
    )
    ham_acc = test(
        start=h_stop,
        dir=h_dir,
        vocab=vocab,
        i_vocab=i_vocab,
        params=params,
        cls="ham",
        model=model,
    )
    print(f"spam acc: {spam_acc*100:.2f}%")
    print(f"ham acc:  {ham_acc*100:.2f}%")


class TreeNode:
    def __init__(self, feature=None, thresh=None, left=None, right=None, label=None):
        self.feature = feature
        self.thresh = thresh
        self.left = left
        self.right = right
        self.label = label


def fit_median_tree(mat, labels, max_depth=10, min_size=5, depth=0, random_split=False):
    uniq = set(labels)
    if len(uniq) == 1 or len(labels) <= min_size or depth >= max_depth:
        return TreeNode(label=max(uniq, key=labels.count))

    n, V = mat.shape

    # initialize best split
    best_j = best_left = best_right = None
    best_thresh = 0.0
    best_mis = n + 1

    if random_split:
        # random feature (deterministic here), but still split on median
        best_j = depth % V
        best_thresh = float(np.median(mat[:, best_j]))
        mask = mat[:, best_j] <= best_thresh
        best_left, best_right = np.nonzero(mask)[0], np.nonzero(~mask)[0]
        if not best_left.size or not best_right.size:
            best_j = None  # fall back to leaf
    else:
        # brute‐force every feature
        labels_arr = np.array(labels)
        for j in range(V):
            col = mat[:, j]
            thresh = float(np.median(col))
            mask = col <= thresh
            left_idx = np.nonzero(mask)[0]
            right_idx = np.nonzero(~mask)[0]
            if left_idx.size == 0 or right_idx.size == 0:
                continue

            # misclassification count via bincount
            left_lbls = labels_arr[left_idx]
            right_lbls = labels_arr[right_idx]
            c0, c1 = np.bincount(left_lbls, minlength=2)
            mis_left = left_lbls.size - max(c0, c1)
            c0, c1 = np.bincount(right_lbls, minlength=2)
            mis_right = right_lbls.size - max(c0, c1)
            mis = mis_left + mis_right

            if mis < best_mis:
                best_mis, best_j, best_thresh = mis, j, thresh
                best_left, best_right = left_idx, right_idx

    # if no split, make leaf
    if best_j is None:
        return TreeNode(label=max(uniq, key=labels.count))

    # recurse
    Lm, Rm = mat[best_left], mat[best_right]
    Ll = [labels[i] for i in best_left]
    Rl = [labels[i] for i in best_right]
    left_node = fit_median_tree(Lm, Ll, max_depth, min_size, depth + 1, random_split)
    right_node = fit_median_tree(Rm, Rl, max_depth, min_size, depth + 1, random_split)

    return TreeNode(
        feature=best_j, thresh=best_thresh, left=left_node, right=right_node
    )


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
    right = wrong = 0
    for i in range(start, len(files)):
        fp = os.path.join(dir, files[i])
        if not os.path.isfile(fp):
            continue
        sample_tokens = tok(filepath=fp, vocab=vocab, i_vocab=i_vocab, test=True)[0]
        embedding = build_bow_vector(
            corp=[sample_tokens],
            idx2token=i_vocab,
            vocab=vocab,
            test=True,
            flat=True,
            median=False,
        )
        pred = classify(msg_vec=embedding, params=params, model=model)
        if (pred == 1 and cls == "spam") or (pred == 0 and cls == "ham"):
            right += 1
        else:
            wrong += 1
    return right / (right + wrong) if (right + wrong) > 0 else 0.0


def run_experiments():
    # Build vocab and corpora
    vocab, i_vocab = {}, {}
    spam_corp = tokenize(dir=s_dir, vocab=vocab, i_vocab=i_vocab, stop=s_stop)
    ham_corp = tokenize(dir=h_dir, vocab=vocab, i_vocab=i_vocab, stop=h_stop)

    # Build vectors
    spam_flat = build_bow_vector(
        corp=spam_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=True,
        median=False,
    )
    ham_flat = build_bow_vector(
        corp=ham_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=True,
        median=False,
    )
    spam_docs = build_bow_vector(
        corp=spam_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=False,
        median=False,
    )
    ham_docs = build_bow_vector(
        corp=ham_corp,
        idx2token=i_vocab,
        vocab=vocab,
        test=False,
        flat=False,
        median=False,
    )

    train_spam_mat = np.array(spam_docs)
    train_ham_mat = np.array(ham_docs)
    labels = [1] * len(spam_docs) + [0] * len(ham_docs)
    mat_all = np.vstack([train_spam_mat, train_ham_mat])

    # Build KD-tree and decision tree models
    kd_root = build_kdtree(points=mat_all, labels=labels)
    dt_random_root = fit_median_tree(mat_all, labels, random_split=True)
    dt_nonrand_root = fit_median_tree(mat_all, labels, random_split=False)

    # Number of test samples
    n_spam_test = s_len - s_stop
    n_ham_test = h_len - h_stop

    # Prepare experiments
    experiments = []
    # Naive Bayes
    V = len(vocab)
    alpha = 1
    denom_spam = sum(spam_flat) + alpha * V
    denom_ham = sum(ham_flat) + alpha * V
    params_nb = {
        "spam": {
            "prior": s_stop / (s_len + h_len),
            "theta": [(spam_flat[i] + alpha) / denom_spam for i in range(V)],
        },
        "ham": {
            "prior": h_stop / (s_len + h_len),
            "theta": [(ham_flat[i] + alpha) / denom_ham for i in range(V)],
        },
    }
    experiments.append(("NB", "n_bayes", params_nb))

    # k-NN brute force
    for p in [1, 2, np.inf]:
        experiments.append(
            (
                f"NN brute p={p}",
                "nn",
                {"spam": train_spam_mat, "ham": train_ham_mat, "p": p},
            )
        )
    # k-NN KD-tree
    for p in [1, 2, np.inf]:
        experiments.append((f"NN kd p={p}", "kd_nn", {"root": kd_root, "p": p}))
    # Decision trees
    experiments.append(("DT random", "tree", {"root": dt_random_root}))
    experiments.append(("DT nonrand", "tree", {"root": dt_nonrand_root}))

    results = []
    for name, model, params in experiments:
        start = time.time()
        spam_acc = test(
            start=s_stop,
            dir=s_dir,
            vocab=vocab,
            i_vocab=i_vocab,
            params=params,
            cls="spam",
            model=model,
        )
        ham_acc = test(
            start=h_stop,
            dir=h_dir,
            vocab=vocab,
            i_vocab=i_vocab,
            params=params,
            cls="ham",
            model=model,
        )
        runtime = time.time() - start
        accuracy = (spam_acc * n_spam_test + ham_acc * n_ham_test) / (
            n_spam_test + n_ham_test
        )
        results.append((name, accuracy, runtime))

    # Plot runtime vs accuracy
    fig, ax = plt.subplots()
    for name, acc, rt in results:
        ax.scatter(rt, acc)
        ax.annotate(name, (rt, acc))
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison: Runtime vs Accuracy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiments()

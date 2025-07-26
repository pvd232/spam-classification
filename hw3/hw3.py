import numpy as np
import os

class TreeNode():
    def __init__(self, C: np.ndarray):
        self.cell = C
        self.left = None
        self.right = None
        self.feature = None
        self.t = None
        self.label = None
        self.leaf = False


def print_tree(node: TreeNode, depth=0):
    """
    Recursively prints the decision tree structure.

    Args:
        node (TreeNode): the current node to print.
        depth (int): current depth (used for indentation).
    """
    if node is None:
        return

    indent = "    " * depth
    # Print this node’s info; adjust to show whatever you like.
    print(
        f"{indent}Node(f={node.feature}, t={node.t:.3f} shape={node.cell.shape}, label={node.label})"
    )

    # Recurse on children
    if node.left or node.right:
        print(f"{indent}├─ left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}└─ right:")
        print_tree(node.right, depth + 1)


def H(col: list[int]) -> float:        
    # 1) ensure an integer array
    col = np.asarray(col, dtype=int)
    K = np.max(col) + 1
    
    # 2) fast bincount
    counts = np.bincount(col, minlength=K)
    ps = counts / counts.sum()

    # 3) omit zero‑probability terms to avoid log(0)
    ps_nonzero = ps[ps > 0]

    # 4) compute entropy
    return -np.sum(ps_nonzero * np.log(ps_nonzero))


def train_tree(node:TreeNode):
    n, d = node.cell.shape

    ys = node.cell[:, d - 1].astype(int)
    K = np.max(ys) + 1

    node.label = int(np.bincount(ys, minlength=K).argmax())
    if n == 1 or H(ys) == 0:        
        node.feature = -1
        node.leaf = True
        return

    best_left = best_right = None
    best_IG = float("-inf")
    best_t_global = None

    # extract entropy of current cell to avoid repeated calculations
    curr_H = H(ys)

    # increment across each feature to find best IG
    f_split = None
    for j in range(d-1):
        # Get list of row indices that would result from sorting the matrix by the feature col
        idxs = np.argsort(node.cell[:, j])

        # Get labels row sorted according to feature-sorted indices from prev step
        labels_sorted = ys[idxs]

        # Initialize left branch to be empty and right with all the labels
        left_cnts = np.zeros(K, dtype=int)
        rt_cnts = np.bincount(labels_sorted, minlength=K)

        col_IG = float("-inf")
        best_t = 0

        # BF increment across each feature value to find best thresh for feature
        col_sorted = node.cell[idxs, j]
        # Increment through each sample
        for i in range(n-1):
            lbl = labels_sorted[i]

            # move labels from right → left
            rt_cnts[lbl] -=1
            left_cnts[lbl] += 1

            # Don't split on homogenous feature values
            if col_sorted[i] == col_sorted[i+1]:
                continue

            # Candidate threshold
            t = (col_sorted[i] + col_sorted[i+1]) / 2

            # Size of each branch
            n_l = i+1
            n_r = n - n_l

            # Frequency of each class label in each branch
            p_l:list[float] = left_cnts / n_l
            p_r:list[float] = rt_cnts / n_r

            # Boolean mask
            nz_l:list[bool] = p_l > 0
            nz_r: list[bool] = p_r > 0

            # Entropy
            H_l = -np.sum(p_l[nz_l] * np.log(p_l[nz_l]))
            H_r = -np.sum(p_r[nz_r] * np.log(p_r[nz_r]))

            w_l_IG = (n_l / n) * H_l
            w_r_IG = (n_r / n) * H_r
            curr_col_IG = curr_H - w_l_IG - w_r_IG

            if curr_col_IG > col_IG:
                col_IG = curr_col_IG                
                best_t = t

        # Compare IG of curr feature decision boundary to best feature decision boundary IG
        if col_IG > best_IG:
            best_IG = col_IG

            # Take the feature‑column j of your data, but already permuted into sorted order by idxs
            # Compare each of those sorted values to the chosen threshold and return array of booleans
            # mask == [ True, True, True, False, False ]
            mask = col_sorted <= best_t

            # Slice matrix according to idxs up until mask
            best_left  = node.cell[idxs[mask]]

            # Slice matrix according to idxs after mask
            # ~mask is the element‑wise NOT of the boolean array, so it picks the False positions
            best_right = node.cell[idxs[~mask]]
            best_t_global = best_t
            f_split = j
            
    if best_IG <= 0 or best_left.shape[0] == 0 or best_right.shape[0] == 0:
        node.feature = -1
        node.leaf = True
        return

    # After you’ve split into best_left and best_right:
    cell_l_y = best_left[:, d-1].astype(int)
    cell_r_y = best_right[:, d-1].astype(int)

    # Compute majority label for left & right
    # c_l and c_r are arrays of counts per class in left and right child:
    c_l = np.bincount(cell_l_y, minlength=K)
    c_r = np.bincount(cell_r_y, minlength=K)

    # Total counts of each class in this node:
    total = c_l + c_r

    # Label is the class with the larger total count
    node.label = int(total.argmax())

    node.feature = f_split
    node.t = float(best_t_global)
    node.left = TreeNode(best_left)
    node.right = TreeNode(best_right)
    train_tree(node=node.left)
    train_tree(node=node.right)

def classify(x:list[int], root: TreeNode) -> int:    
    curr = root
    while not curr.leaf:        
        if float(x[curr.feature]) <= curr.t:
            curr = curr.left
        else:
            curr = curr.right
    return curr.label


def test_tree(root: TreeNode, test_data:np.ndarray):    
    t = 0
    f = 0
    for i in range(test_data.shape[0]):
        x_test = test_data[i]
        res = classify(x=x_test, root=root)
        if res == int(x_test[-1]):
            t += 1
        else:
            f += 1
    print(f"f {round(100*(f/(f+t)))}%\nt {round(100*(t/(f+t)))}% ")


def pre_process(filepath: str) -> list[str]:
    # Try utf‑8, fall back to latin‑1
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()
    tokens = text.split("\n")
    # Remove trailing white space
    tokens = tokens[:-1]
    X = np.array([x.split(",") for x in tokens])
    
    # Reformat to chars to floats
    X = X.astype(float)
    return X


def main():    
    data_dir = os.path.join(os.curdir, "spambase/spambase.data")
    X = pre_process(data_dir)    
    
    # Shuffle the data
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    tr_cov = 0.8  # train / test split
    stop = round(tr_cov * X.shape[0])
    train_X = X[:stop, :]
    test_X = X[stop:, :]
    
    root = TreeNode(train_X)    
    train_tree(node=root)
    test_tree(root=root,test_data=test_X)

if __name__ == "__main__":
    main()

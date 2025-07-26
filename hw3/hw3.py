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
    print(f"{indent}Node(f={node.feature}, shape={node.cell.shape}, label={node.label})")

    # Recurse on children
    if node.left or node.right:
        print(f"{indent}├─ left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}└─ right:")
        print_tree(node.right, depth + 1)


def H(col: list[int]) -> float:        
    # 1) ensure an integer array
    col = np.asarray(col, dtype=int)

    # 2) fast bincount
    counts = np.bincount(col, minlength=2)
    ps = counts / counts.sum()

    # 3) omit zero‑probability terms to avoid log(0)
    ps_nonzero = ps[ps > 0]

    # 4) compute entropy
    return -np.sum(ps_nonzero * np.log(ps_nonzero))

def IG(h:float,cl_col:list[int], cr_col:list[int]) -> float:
    n = len(cl_col) + len(cr_col)
    w_l = len(cl_col) / n
    w_r = len(cr_col) / n
    return h - w_l*H(cl_col) - w_r*H(cr_col)


def train_tree(node:TreeNode):
    n, d = node.cell.shape

    ys = node.cell[:, d - 1].astype(int)
    node.label = int(np.bincount(ys, minlength=2).argmax())
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

        # sort data by selected feature
        idxs = np.argsort(node.cell[:, j])

        # extract feature col
        col_IG = float("-inf")
        best_t = 0

        # BF increment across each feature value to find best thresh for feature
        col_sorted = node.cell[idxs, j]

        # skip feature if 0 info gain
        if np.all(col_sorted == col_sorted[0]):
            continue

        labels_sorted = node.cell[idxs, -1]
        
        # Pick threshold in-between feature values to avoid ties
        values = np.unique(col_sorted)
        threshes = (values[:-1] + values[1:]) / 2
        for t in threshes:                    
            m = np.searchsorted(col_sorted, t, side="right")            
            tmp_l_labels = labels_sorted[:m]
            tmp_r_labels = labels_sorted[m:]
            curr_col_IG = IG(h=curr_H,cl_col=tmp_l_labels,cr_col=tmp_r_labels)
            if curr_col_IG > col_IG:
                col_IG = curr_col_IG                
                best_t = t

        # compare IG of curr feature decision boundary to best feature decision boundary IG
        if col_IG > best_IG:
            best_IG = col_IG
            sorted_mat = node.cell[idxs]            
            best_left = sorted_mat[sorted_mat[:, j] <= best_t]            
            best_right = sorted_mat[sorted_mat[:, j] > best_t]
            best_t_global = best_t
            f_split = j
    if best_IG <= 0 or best_left.shape[0] == 0 or best_right.shape[0] == 0:
        node.feature = -1
        node.leaf = True
        return

        # after you’ve split into best_left and best_right:
    cell_l_y = best_left[:, d-1].astype(int)
    cell_r_y = best_right[:, d-1].astype(int)

    # compute majority label for left & right
    # c_l and c_r are arrays of counts per class in left and right child:
    c_l = np.bincount(cell_l_y, minlength=2)
    c_r = np.bincount(cell_r_y, minlength=2)

    # total counts of each class in this node:
    total = c_l + c_r

    # label is the class with the larger total count
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

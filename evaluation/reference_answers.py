"""
evaluation/reference_answers.py
────────────────────────────────
Gold-standard reference answers for common DSA topics used as ROUGE baselines.

Each entry is a dict with:
    topic            (str)   — canonical topic name
    level            (int)   — ALPS difficulty level 1–4
    reference_answer (str)   — concise but complete reference explanation

These are used by evaluate_response() in metrics.py.  When a new topic is
added to the app, add a matching entry here so automated evaluation works.
"""

REFERENCE_ANSWERS: list[dict] = [
    {
        "topic": "binary search",
        "level": 2,
        "reference_answer": (
            "Binary search is an efficient algorithm for finding a target value "
            "in a sorted array. It works by repeatedly halving the search space: "
            "compare the target to the middle element, then discard the half that "
            "cannot contain the target. If the middle element equals the target, "
            "the search succeeds; if the target is smaller, search the left half; "
            "if larger, search the right half. This continues until the element is "
            "found or the search space is empty. Binary search runs in O(log n) "
            "time because each comparison eliminates half the remaining elements. "
            "It requires the array to be sorted beforehand. The space complexity "
            "is O(1) for the iterative version and O(log n) for the recursive "
            "version due to the call stack."
        ),
    },
    {
        "topic": "linked lists",
        "level": 1,
        "reference_answer": (
            "A linked list is a linear data structure where elements, called nodes, "
            "are stored in memory non-contiguously. Each node contains two parts: "
            "the data value and a pointer (or reference) to the next node in the "
            "sequence. The list starts at a head node; the last node points to null "
            "to mark the end. In a singly linked list each node points only forward. "
            "In a doubly linked list each node has both a next and a previous pointer, "
            "allowing traversal in both directions. Linked lists support O(1) insertion "
            "and deletion at the head, but accessing an element by index requires O(n) "
            "traversal from the head because there is no direct index lookup like in an "
            "array. Common operations include traversal, insertion, deletion, and "
            "reversing the list."
        ),
    },
    {
        "topic": "recursion",
        "level": 2,
        "reference_answer": (
            "Recursion is a programming technique where a function calls itself to "
            "solve smaller subproblems of the same kind. Every recursive function "
            "must have two parts: a base case that stops the recursion by returning "
            "a result directly, and a recursive case that breaks the problem into a "
            "smaller instance and calls itself. Without a base case the function "
            "recurses infinitely until a stack overflow occurs. Each function call is "
            "placed on the call stack, which stores local variables and the return "
            "address. When the base case is reached, calls resolve back up the stack "
            "in reverse order. A classic example is computing factorial: "
            "factorial(n) = n * factorial(n-1) with factorial(0) = 1 as the base case. "
            "Recursion is naturally suited to problems with tree or divide-and-conquer "
            "structure such as tree traversal, merge sort, and the Fibonacci sequence. "
            "The time and space complexity depends on the depth and branching factor "
            "of the recursive calls."
        ),
    },
    {
        "topic": "sorting algorithms",
        "level": 2,
        "reference_answer": (
            "Sorting algorithms arrange elements in a specific order, typically "
            "ascending or descending. Common algorithms and their complexities: "
            "Bubble sort repeatedly swaps adjacent elements that are out of order, "
            "O(n^2) time in the worst case. Selection sort finds the minimum element "
            "and places it at the start of the unsorted portion, also O(n^2). "
            "Insertion sort builds a sorted subarray one element at a time, O(n^2) "
            "worst case but O(n) when nearly sorted. Merge sort divides the array "
            "in half, recursively sorts each half, then merges them in O(n log n) "
            "time with O(n) auxiliary space. Quick sort picks a pivot and partitions "
            "elements around it, achieving O(n log n) average time but O(n^2) in the "
            "worst case (mitigated with random pivot selection). Heap sort uses a "
            "max-heap structure and runs in O(n log n) time with O(1) space. "
            "For small or nearly sorted arrays, insertion sort is often fastest in "
            "practice despite its worst-case complexity."
        ),
    },
    {
        "topic": "hash tables",
        "level": 2,
        "reference_answer": (
            "A hash table (also called a hash map) is a data structure that maps "
            "keys to values using a hash function. The hash function converts a key "
            "into an integer index that determines where the corresponding value is "
            "stored in an underlying array (the bucket array). Lookup, insertion, and "
            "deletion are all O(1) on average because the hash function computes the "
            "index directly without scanning. Collisions occur when two different keys "
            "hash to the same index. Common collision resolution strategies are "
            "chaining (each bucket holds a linked list of entries) and open addressing "
            "(probe for the next available slot using linear probing, quadratic probing, "
            "or double hashing). The load factor is the ratio of stored entries to the "
            "total number of buckets; when it exceeds a threshold (commonly 0.75) the "
            "table is resized and all entries are rehashed. In the worst case (many "
            "collisions) performance degrades to O(n). Hash tables are the underlying "
            "structure for Python dicts and sets."
        ),
    },
    {
        "topic": "arrays",
        "level": 1,
        "reference_answer": (
            "An array is a data structure that stores elements of the same type in "
            "contiguous memory locations. Each element is accessed directly using a "
            "zero-based index, so reading or writing arr[i] takes O(1) time because "
            "the address can be computed directly from the index. In most languages "
            "arrays have a fixed size set at creation, so growing them typically means "
            "allocating a new, larger array and copying elements over. Inserting or "
            "deleting an element at an arbitrary position requires shifting all "
            "subsequent elements to keep the array contiguous, which takes O(n) time "
            "in the worst case. Traversing every element takes O(n) time. Searching "
            "for a value takes O(n) time if the array is unsorted, but only O(log n) "
            "time with binary search if the array is sorted. A simple example in "
            "C-like syntax is int arr[] = {1, 2, 3}, which declares an array holding "
            "the three integers 1, 2, and 3 at indices 0, 1, and 2."
        ),
    },
    {
        "topic": "binary trees",
        "level": 3,
        "reference_answer": (
            "A binary tree is a hierarchical data structure in which each node has "
            "at most two children, referred to as the left child and the right child. "
            "The topmost node is the root, and nodes with no children are called "
            "leaves. In a balanced binary tree the height is O(log n), since the "
            "number of nodes roughly doubles at each level, but a degenerate (skewed) "
            "tree can have height O(n). The three standard traversals — inorder "
            "(left, root, right), preorder (root, left, right), and postorder "
            "(left, right, root) — each visit every node exactly once and run in "
            "O(n) time. A binary search tree (BST) adds an ordering property: for "
            "every node, all values in the left subtree are less than the node's "
            "value and all values in the right subtree are greater. This property "
            "allows search, insertion, and deletion to run in O(log n) time on "
            "average, because each comparison eliminates roughly half of the "
            "remaining nodes, similar to binary search. In the worst case, however "
            "— when the tree is unbalanced or skewed into a linked-list shape — "
            "these operations degrade to O(n). Binary trees underlie many "
            "applications, including expression trees for parsing arithmetic "
            "expressions and BSTs for maintaining sorted, searchable collections."
        ),
    },
    {
        "topic": "graphs",
        "level": 3,
        "reference_answer": (
            "A graph is a data structure consisting of a set of V vertices (nodes) "
            "connected by a set of E edges. Graphs can be directed, where edges have "
            "a direction from one vertex to another, or undirected, where edges "
            "represent a two-way connection; they can also be weighted, where each "
            "edge carries a cost or distance, or unweighted. Two common "
            "representations are the adjacency matrix, a V×V grid indicating which "
            "vertex pairs are connected, using O(V^2) space, and the adjacency list, "
            "where each vertex stores a list of its neighbors, using O(V+E) space — "
            "more efficient for sparse graphs. Breadth-first search (BFS) explores a "
            "graph level by level using a queue and finds the shortest path between "
            "two vertices in an unweighted graph, running in O(V+E) time. "
            "Depth-first search (DFS) explores as far as possible along each branch "
            "before backtracking, implemented with a stack or recursion, and also "
            "runs in O(V+E) time. Graphs model a wide range of real-world problems, "
            "including social networks (people as vertices, friendships as edges), "
            "routing and navigation systems (locations as vertices, roads as "
            "weighted edges), and dependency resolution such as build systems or "
            "course prerequisites, where a topological sort orders vertices "
            "consistently with directed edges."
        ),
    },
    {
        "topic": "dynamic programming",
        "level": 4,
        "reference_answer": (
            "Dynamic programming (DP) is an optimization technique that improves on "
            "naive recursion by storing, or 'memoizing,' the results of subproblems "
            "so they are never recomputed. A problem is a good candidate for DP when "
            "it has optimal substructure, meaning an optimal solution can be built "
            "from optimal solutions to its subproblems, and overlapping "
            "subproblems, meaning the same subproblems recur many times during a "
            "naive recursive solution. There are two standard approaches: top-down "
            "memoization, which keeps the natural recursive structure but caches "
            "each subproblem's result the first time it is computed, and bottom-up "
            "tabulation, which instead builds up an iterative table of subproblem "
            "results starting from the smallest subproblems. Both approaches "
            "typically reduce time complexity from exponential to polynomial by "
            "ensuring each distinct subproblem is solved only once. Classic examples "
            "include computing the nth Fibonacci number in O(n) time instead of "
            "exponential time, the 0/1 knapsack problem in O(nW) time where n is the "
            "number of items and W is the capacity, and the longest common "
            "subsequence of two strings in O(mn) time where m and n are the string "
            "lengths. When only the immediately preceding row or a small window of "
            "previous results is needed to compute the next one, the space "
            "complexity can often be reduced from O(n) or O(nW) down to O(1) or "
            "O(W) by discarding earlier rows once they are no longer needed."
        ),
    },
]

# Convenience lookup: topic string -> entry dict
REFERENCE_BY_TOPIC: dict[str, dict] = {
    entry["topic"]: entry for entry in REFERENCE_ANSWERS
}

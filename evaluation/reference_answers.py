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
]

# Convenience lookup: topic string -> entry dict
REFERENCE_BY_TOPIC: dict[str, dict] = {
    entry["topic"]: entry for entry in REFERENCE_ANSWERS
}

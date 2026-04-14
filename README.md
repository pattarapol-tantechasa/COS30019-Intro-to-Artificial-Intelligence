# COS30019-Intro-to-Artificial-Intelligence

## How to run search?

Format : `python search.py tests/<test_file> <method>`

```bash
# Change Directory
cd 2A-tree-based-search

# Search with BFS
python search.py tests/PathFinder-test.txt BFS

# Search with DFS
python search.py tests/PathFinder-test.txt DFS
```

## How to run Test Runner

There are two ways to use `test_runner.py`
1. Use this test runner to run specific test with all algorithm
   - `python test_runner.py <test_file_name>`
2. Use test runner to run all test for specific algorithm
   - `python test_runner.py --algo <algorithm_name>`

```bash
# Change Directory
cd 2A-tree-based-search

# Run test5 with all algorithm
python test_runner.py test5

# Run all test for BFS
python test_runner.py --algo BFS
```
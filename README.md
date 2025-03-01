# code_verify

A Python library for evaluating the similarity between code implementations. Provides multiple approaches to comparing code for different use cases.

### Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/andthattoo/code_verify.git

# Or clone and install locally
git clone https://github.com/andthattoo/code_verify.git
cd code_verify
pip install -e .
```

### Features

#### Text-based similarity
Uses difflib to calculate edit distance between code snippets. This approach is fast and straightforward but sensitive to formatting, variable names, and implementation details.

```python
from code_verify import calculate_code_similarity

similarity, metadata = calculate_code_similarity(
    ground_truth_code, 
    predicted_code, 
    use_ast=False
)

print(f"Similarity: {similarity}")
```

#### AST-based similarity
Compares the Abstract Syntax Tree structure of code, making it resilient to differences in variable names, formatting, and certain implementation variations.

```python
from code_verify import calculate_code_similarity

similarity, metadata = calculate_code_similarity(
    ground_truth_code,
    predicted_code,
    use_ast=True
)

print(f"AST similarity: {metadata['ast_similarity']}")
```

#### Execution-based similarity
Evaluates code by executing it with test cases and comparing outputs, identifying functional equivalence regardless of implementation.

```python
from code_verify import calculate_execution_similarity, generate_test_cases

# Generate test cases automatically based on function signature
test_cases = generate_test_cases("factorial", ["n"], count=5)

# Compare execution results
similarity, metadata = calculate_execution_similarity(
    ground_truth_code,
    predicted_code,
    test_cases=test_cases
)

print(f"Execution similarity: {similarity}")
```

#### Patch-based similarity
For comparing code changes (patches) rather than complete implementations:

```python
from code_verify import calculate_change_similarity

similarity, metadata = calculate_change_similarity(
    code_context,           # Original files
    oracle_new_content,     # Expected content after changes
    predicted_new_content   # Actual content after changes
)
```

### Safety

Execution-based similarity includes protections:
- Sandbox environment that blocks dangerous operations
- Execution timeouts
- Multiple execution attempts for stability

### Examples

The package includes comprehensive examples demonstrating each approach:

```python
from code_verify.examples import (
    example_basic_usage,
    example_multiple_changes,
    example_flexible_tags,
    example_code_similarity,
    example_ast_similarity,
    example_change_similarity,
    example_execution_similarity
)

# Run all examples
example_execution_similarity()
```

## License

MIT
"""
Examples demonstrating the usage of code_verify library.
"""

from code_verify import (
    calculate_search_replace_similarity,
    extract_thought_solution,
    calculate_code_similarity,
    calculate_change_similarity,
    calculate_execution_similarity,
    generate_test_cases,
    generate_unified_diff,
    DEFAULT_EXEC_TIMEOUT
)


def example_basic_usage():
    """Basic example demonstrating the core functionality."""
    # Original code files
    code_context = {
        "src/example.py": "def add(a, b):\n    return a + b\n"
    }

    # Expected (oracle) code after changes
    oracle_new_content = {
        "src/example.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n"
    }

    # AI's output with thought process and solution
    ai_output = """<think>
I need to add a comment to the add function to explain what it does.
</think>
<solution>
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
</solution>"""

    # Calculate reward
    reward, metadata = calculate_search_replace_similarity(
        code_context, oracle_new_content, ai_output
    )

    print(f"Reward: {reward}")
    print(f"Metadata: {metadata}")
    return reward, metadata


def example_multiple_changes():
    """Example with multiple files and changes."""
    # Original code files
    code_context = {
        "src/math.py": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        "src/utils.py": "def greet(name):\n    return f'Hello, {name}!'\n"
    }

    # Expected (oracle) code after changes
    oracle_new_content = {
        "src/math.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n\ndef subtract(a, b):\n    # Subtract b from a\n    return a - b\n",
        "src/utils.py": "def greet(name):\n    # Greet a person\n    return f'Hello, {name}!'\n"
    }

    # AI's output with thought process and solution
    ai_output = """<think>
I need to add docstrings to all functions to make the code more readable.
</think>
<solution>
```python
### src/math.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```

```python
### src/math.py
<<<<<<< SEARCH
def subtract(a, b):
    return a - b
=======
def subtract(a, b):
    # Subtract b from a
    return a - b
>>>>>>> REPLACE
```

```python
### src/utils.py
<<<<<<< SEARCH
def greet(name):
    return f'Hello, {name}!'
=======
def greet(name):
    # Greet a person
    return f'Hello, {name}!'
>>>>>>> REPLACE
```
</solution>"""

    # Calculate reward
    reward, metadata = calculate_search_replace_similarity(
        code_context, oracle_new_content, ai_output
    )

    print(f"Reward: {reward}")
    print(f"Metadata: {metadata}")
    return reward, metadata


def example_flexible_tags():
    """Example demonstrating flexible tag handling."""
    # Original code files
    code_context = {
        "src/example.py": "def add(a, b):\n    return a + b\n"
    }

    # Expected code after changes
    oracle_new_content = {
        "src/example.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n"
    }

    # Different tag formats
    outputs = [
        # Standard format
        """<think>
I need to add a comment to the add function.
</think>
<solution>
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
</solution>""",

        # Alternative thinking tags
        """<reasoning>
I need to add a comment to the add function.
</reasoning>
<solution>
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
</solution>""",

        # Alternative answer tags
        """<think>
I need to add a comment to the add function.
</think>
<answer>
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
</answer>""",

        # Markdown-style tags
        """# Thinking
I need to add a comment to the add function.
# End Thinking

# Solution
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
# End Solution""",

        # No thinking section
        """<solution>
```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```
</solution>""",

        # No tags at all, just search/replace blocks
        """```python
### src/example.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    # Add two numbers
    return a + b
>>>>>>> REPLACE
```"""
    ]

    for i, output in enumerate(outputs):
        print(f"\nExample {i+1}:")
        # Extract thought and solution to demonstrate
        thought, answer = extract_thought_solution(output, require_thinking=False)
        print(f"Thought found: {thought is not None}")
        if thought:
            print(f"Thought: {thought[:50]}..." if len(thought) > 50 else f"Thought: {thought}")
        print(f"Answer found: {answer is not None}")
        
        # Calculate reward
        reward, metadata = calculate_search_replace_similarity(
            code_context, oracle_new_content, output, require_thinking=False
        )
        print(f"Reward: {reward}")


def example_code_similarity():
    """Example demonstrating the code similarity feature."""
    # Expected code
    expected_code = """def factorial(n):
    # Calculate the factorial of n.
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
"""

    # Different variations of generated code
    generated_codes = [
        # Perfect match
        """def factorial(n):
    # Calculate the factorial of n.
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
""",
        # Slight variation (different variable name)
        """def factorial(n):
    # Calculate the factorial of n.
    if n == 0 or n == 1:
        return 1
    else:
        product = 1
        for i in range(2, n + 1):
            product *= i
        return product
""",
        # Different implementation (recursive)
        """def factorial(n):
    # Calculate the factorial of n.
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
""",
        # With bugs
        """def factorial(n):
    # Calculate the factorial of n.
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n):  # Bug: Missing +1 
            result *= i
        return result
"""
    ]

    for i, generated_code in enumerate(generated_codes):
        print(f"\nCode variant {i+1}:")
        similarity, metadata = calculate_code_similarity(
            expected_code, generated_code
        )
        print(f"Similarity: {similarity:.4f}")
        print(f"Changes in diff: {'None' if not metadata['diff'] else '(see below)'}")
        if metadata['diff']:
            print(f"Diff:\n{metadata['diff']}")


def example_ast_similarity():
    """Example demonstrating the AST-based similarity feature."""
    ground_truth_code = """
def process_data(items):
    result = []
    for item in items:
        if item > 10:
            result.append(item * 2)
    return result
"""

    # Functionally equivalent but different variable names and formatting
    predicted_code = """
def analyze_list(data):
    output = []
    for element in data:
        if element > 10:
            output.append(element * 2)
    return output
"""

    # Compare using both text and AST similarity
    similarity, metadata = calculate_code_similarity(
        ground_truth_code, predicted_code, use_ast=True
    )
    
    # Compare using only text similarity
    text_similarity, text_metadata = calculate_code_similarity(
        ground_truth_code, predicted_code, use_ast=False
    )
    
    print("\nComparing structurally similar code with different variable names:")
    print(f"Combined similarity score: {similarity:.4f}")
    print(f"Text-only similarity: {text_metadata['text_similarity']:.4f}")
    print(f"AST-only similarity: {metadata['ast_similarity']:.4f}")
    print("\nThis demonstrates how AST similarity captures structural/functional")
    print("similarity even when variable names and formatting differ.")
    print("\nDiff showing the textual differences:")
    print(metadata['diff'])
    
    return similarity, metadata


def example_change_similarity():
    """Example demonstrating code change similarity calculation."""
    # Original code files
    code_context = {
        "src/user.py": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def display(self):
        return f"{self.name} ({self.email})"
"""
    }

    # Oracle (expected) changes
    oracle_new_content = {
        "src/user.py": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True
    
    def display(self):
        status = "active" if self.active else "inactive"
        return f"{self.name} ({self.email}) - {status}"
        
    def deactivate(self):
        self.active = False
"""
    }

    # Different predicted changes with varying accuracy
    predicted_changes = [
        # Perfect match
        {
            "src/user.py": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True
    
    def display(self):
        status = "active" if self.active else "inactive"
        return f"{self.name} ({self.email}) - {status}"
        
    def deactivate(self):
        self.active = False
"""
        },
        # Good match but different implementation details
        {
            "src/user.py": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True
    
    def display(self):
        return f"{self.name} ({self.email}) - {'active' if self.active else 'inactive'}"
        
    def deactivate(self):
        self.active = False
"""
        },
        # Missing deactivate method
        {
            "src/user.py": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True
    
    def display(self):
        status = "active" if self.active else "inactive"
        return f"{self.name} ({self.email}) - {status}"
"""
        }
    ]

    print("\nEvaluating different code change implementations:")
    for i, pred_new_content in enumerate(predicted_changes):
        print(f"\nImplementation {i+1}:")
        similarity, metadata = calculate_change_similarity(
            code_context, oracle_new_content, pred_new_content
        )
        print(f"Change similarity score: {similarity:.4f}")
        print("File-by-file similarities:")
        for sim in metadata["similarities"]:
            print(f"- {sim['path']}: {sim['similarity']:.4f}")
            if sim["pred_change"] and sim["oracle_change"]:
                print(f"  Changes detected in both oracle and prediction")
    
    return similarity, metadata




def example_execution_similarity():
    """Example demonstrating execution-based similarity."""
    # Sample problem: Calculate factorial
    ground_truth_code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
"""

    # Various implementations to test
    implementations = [
        # Identical implementation
        ground_truth_code,
        
        # Recursive implementation 
        """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
"""
        ,
        
        # Different parameter name
        """
def factorial(number):
    if number == 0 or number == 1:
        return 1
    else:
        result = 1
        for i in range(2, number + 1):
            result *= i
        return result
"""
        ,
        
        # Buggy implementation - off by one
        """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        result = 1
        # Bug: Missing +1 in range end
        for i in range(2, n):
            result *= i
        return result
"""
        ,
        
        # Completely different algorithm (returns n^2 instead)
        """
def factorial(n):
    return n * n
"""
    ]
    
    print("\nComparing different factorial implementations using execution:")
    
    # Generate some test cases
    test_cases = generate_test_cases("factorial", ["n"], count=5)
    
    # Set specific test cases for clarity
    override_cases = [
        {"inputs": {"n": 0}, "expected_output": None},
        {"inputs": {"n": 1}, "expected_output": None},
        {"inputs": {"n": 5}, "expected_output": None},
        {"inputs": {"n": 7}, "expected_output": None},
        {"inputs": {"n": 10}, "expected_output": None},
    ]
    
    # Compare all implementations
    for i, implementation in enumerate(implementations):
        print(f"\nImplementation {i+1}:")
        
        # Run all similarity metrics
        execution_similarity, execution_metadata = calculate_execution_similarity(
            ground_truth_code, implementation, test_cases=override_cases
        )
        
        text_similarity, _ = calculate_code_similarity(
            ground_truth_code, implementation, use_ast=False, use_execution=False
        )
        
        combined_similarity, combined_metadata = calculate_code_similarity(
            ground_truth_code, implementation, 
            use_ast=True, 
            use_execution=True,
            test_cases=override_cases
        )
        
        print(f"Execution similarity: {execution_similarity:.4f}")
        print(f"Text similarity: {text_similarity:.4f}")
        if combined_metadata.get("ast_valid", False):
            print(f"AST similarity: {combined_metadata.get('ast_similarity', 0):.4f}")
        print(f"Combined similarity: {combined_similarity:.4f}")
        
        # Show execution details for the first test case
        if execution_metadata.get("execution_valid", False):
            print("\nTest case results:")
            success_count = 0
            for idx, test_result in enumerate(execution_metadata["test_results"]):
                inputs = test_result["inputs"]
                output_match = test_result.get("outputs_match", False)
                print(f"  Test {idx+1}: input={inputs}, match={output_match}")
                if output_match:
                    success_count += 1
            
            print(f"\nSuccessful tests: {success_count}/{len(execution_metadata['test_results'])}")
            
            if i > 0:  # Skip diff for identical implementations
                print("\nDiff:")
                text_diff = generate_unified_diff(ground_truth_code, implementation)
                print(text_diff)
    
    return execution_similarity, execution_metadata


if __name__ == "__main__":
    print("Running basic example:")
    example_basic_usage()
    
    print("\nRunning multiple changes example:")
    example_multiple_changes()
    
    print("\nRunning flexible tags example:")
    example_flexible_tags()
    
    print("\nRunning code similarity example:")
    example_code_similarity()
    
    print("\nRunning AST similarity example:")
    example_ast_similarity()
    
    print("\nRunning change similarity example:")
    example_change_similarity()
    
    print("\nRunning execution similarity example:")
    example_execution_similarity()
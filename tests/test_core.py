import unittest
from code_verify import (
    FormatError,
    extract_thought_solution,
    parse_search_replace,
    generate_unified_diff,
    apply_code_change,
    calculate_change_similarity,
    calculate_search_replace_similarity,
    calculate_code_similarity,
)


class TestCodeVerify(unittest.TestCase):
    def test_extract_thought_solution(self):
        # Standard format
        output = """<think>
This is my thought.
</think>
<solution>
This is my solution.
</solution>"""
        thought, solution = extract_thought_solution(output)
        self.assertEqual(thought, "This is my thought.")
        self.assertEqual(solution, "This is my solution.")

        # Alternative tags
        output = """<reasoning>
This is my reasoning.
</reasoning>
<answer>
This is my answer.
</answer>"""
        thought, solution = extract_thought_solution(output, require_thinking=False)
        self.assertEqual(thought, "This is my reasoning.")
        self.assertEqual(solution, "This is my answer.")

        # Another XML tag style
        output = """<rationale>
This is my thought process.
</rationale>

<result>
This is the solution.
</result>"""
        thought, solution = extract_thought_solution(output, require_thinking=False)
        self.assertEqual(thought, "This is my thought process.")
        self.assertEqual(solution, "This is the solution.")
        
        # Missing thought but not required
        output = """<solution>
This is just the solution.
</solution>"""
        thought, solution = extract_thought_solution(output, require_thinking=False)
        self.assertIsNone(thought)
        self.assertEqual(solution, "This is just the solution.")
        
        # Missing thought when required
        with self.assertRaises(FormatError):
            extract_thought_solution(output, require_thinking=True)
            
        # No tags but contains search/replace blocks
        output = """```python
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
        thought, solution = extract_thought_solution(output, require_thinking=False)
        self.assertIsNone(thought)
        self.assertEqual(solution, output)
        
        # No valid content
        with self.assertRaises(FormatError):
            extract_thought_solution("No tags or search/replace blocks here")

    def test_parse_search_replace(self):
        text = """```python
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
        result = parse_search_replace(text)
        self.assertIn("src/example.py", result)
        self.assertEqual(len(result["src/example.py"]), 1)
        self.assertEqual(result["src/example.py"][0][0], "def add(a, b):\n    return a + b")
        self.assertEqual(result["src/example.py"][0][1], "def add(a, b):\n    # Add two numbers\n    return a + b")

    def test_generate_unified_diff(self):
        old_code = "def add(a, b):\n    return a + b\n"
        new_code = "def add(a, b):\n    # Add two numbers\n    return a + b\n"
        diff = generate_unified_diff(old_code, new_code)
        self.assertIn("+    # Add two numbers", diff)

    def test_apply_code_change(self):
        code_context = {
            "src/example.py": "def add(a, b):\n    return a + b\n"
        }
        search_replace_dict = {
            "src/example.py": [
                ("def add(a, b):\n    return a + b", "def add(a, b):\n    # Add two numbers\n    return a + b")
            ]
        }
        result = apply_code_change(code_context, search_replace_dict)
        self.assertIn("src/example.py", result)
        self.assertEqual(result["src/example.py"], "def add(a, b):\n    # Add two numbers\n    return a + b\n")

        # Test identical search and replace
        with self.assertRaises(FormatError):
            apply_code_change(code_context, {
                "src/example.py": [("def add(a, b):", "def add(a, b):")]
            })

        # Test search block not found
        with self.assertRaises(FormatError):
            apply_code_change(code_context, {
                "src/example.py": [("def multiply(a, b):", "def multiply(a, b, c):")]
            })

    def test_calculate_change_similarity(self):
        code_context = {
            "src/example.py": "def add(a, b):\n    return a + b\n"
        }
        oracle_new_content = {
            "src/example.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n"
        }
        pred_new_content = {
            "src/example.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n"
        }
        similarity, metadata = calculate_change_similarity(code_context, oracle_new_content, pred_new_content)
        self.assertEqual(similarity, 1.0)  # Perfect match

        # Test imperfect match
        pred_new_content_imperfect = {
            "src/example.py": "def add(a, b):\n    # Adds two numbers together\n    return a + b\n"
        }
        similarity, metadata = calculate_change_similarity(code_context, oracle_new_content, pred_new_content_imperfect)
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, 0.0)

    def test_calculate_search_replace_similarity(self):
        code_context = {
            "src/example.py": "def add(a, b):\n    return a + b\n"
        }
        oracle_new_content = {
            "src/example.py": "def add(a, b):\n    # Add two numbers\n    return a + b\n"
        }
        
        # Standard format
        output = """<think>
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
        similarity, metadata = calculate_search_replace_similarity(code_context, oracle_new_content, output)
        self.assertEqual(similarity, 1.0)  # Perfect match
        self.assertIn("thought", metadata)
        self.assertIn("answer", metadata)
        
        # Alternative tags
        output = """<reasoning>
I need to add a comment to the add function to explain what it does.
</reasoning>
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
</answer>"""
        similarity, metadata = calculate_search_replace_similarity(code_context, oracle_new_content, output)
        self.assertEqual(similarity, 1.0)
        self.assertIn("thought", metadata)
        self.assertIn("answer", metadata)
        
        # No thinking section
        output = """<solution>
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
        similarity, metadata = calculate_search_replace_similarity(code_context, oracle_new_content, output, require_thinking=False)
        self.assertEqual(similarity, 1.0)
        self.assertNotIn("thought", metadata)
        self.assertIn("answer", metadata)
        
        # No tags, just the code blocks
        output = """```python
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
        similarity, metadata = calculate_search_replace_similarity(code_context, oracle_new_content, output, require_thinking=False)
        self.assertEqual(similarity, 1.0)
        self.assertNotIn("thought", metadata)
        self.assertIn("answer", metadata)
        
        # Custom tags - let's use XML tags
        output = """<my-thinking>
I need to add a comment to the add function.
</my-thinking>

<my-answer>
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
</my-answer>"""
        similarity, metadata = calculate_search_replace_similarity(
            code_context, 
            oracle_new_content, 
            output,
            thinking_tag="my-thinking",
            answer_tag="my-answer"
        )
        self.assertEqual(similarity, 1.0)
        self.assertIn("thought", metadata)
        self.assertIn("answer", metadata)


    def test_calculate_code_similarity(self):
        # Test identical code
        ground_truth_code = "def add(a, b):\n    return a + b"
        predicted_code = "def add(a, b):\n    return a + b"
        similarity, metadata = calculate_code_similarity(ground_truth_code, predicted_code)
        self.assertEqual(similarity, 1.0)
        self.assertEqual(metadata["diff"], "")
        
        # Test very similar code (whitespace difference)
        predicted_code_whitespace = "def add(a, b):\n    return a+b"
        similarity, metadata = calculate_code_similarity(ground_truth_code, predicted_code_whitespace)
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, 0.9)
        
        # Test substantially different code
        predicted_code_different = "def multiply(a, b):\n    return a * b"
        similarity, metadata = calculate_code_similarity(ground_truth_code, predicted_code_different)
        self.assertLess(similarity, 0.9)  # The threshold is less strict
        
        # Check metadata fields
        self.assertIn("similarity", metadata)
        self.assertIn("diff", metadata)
        self.assertIn("ground_truth_length", metadata)
        self.assertIn("predicted_length", metadata)
        
        # Check AST-based similarity
        self.assertIn("ast_similarity", metadata)
        self.assertIn("ast_valid", metadata)
        self.assertIn("text_similarity", metadata)
        
        # Test with use_ast=False
        similarity_text_only, metadata_text_only = calculate_code_similarity(
            ground_truth_code, predicted_code, use_ast=False
        )
        self.assertEqual(similarity_text_only, 1.0)
        self.assertNotIn("ast_similarity", metadata_text_only)
        
        # Test with non-Python code (should fall back to text similarity)
        non_python_code = "This is not valid Python code!"
        similarity_non_python, metadata_non_python = calculate_code_similarity(
            non_python_code, non_python_code
        )
        self.assertEqual(similarity_non_python, 1.0)  # Text similarity is 1.0
        self.assertEqual(metadata_non_python["ast_valid"], False)
        
        # Test with structurally similar but textually different code
        ground_truth_structured = """
def process_data(items):
    result = []
    for item in items:
        if item > 10:
            result.append(item * 2)
    return result
"""
        predicted_structured = """
def analyze_list(data):
    output = []
    for element in data:
        if element > 20:
            output.append(element * 3)
    return output
"""
        # Should have higher AST similarity than text similarity
        similarity_structured, metadata_structured = calculate_code_similarity(
            ground_truth_structured, predicted_structured
        )
        self.assertGreater(metadata_structured["ast_similarity"], 
                          metadata_structured["text_similarity"])


if __name__ == "__main__":
    unittest.main()
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import ast
import difflib
import re
import sys
import os
import tempfile
import traceback
import inspect
import random
import builtins
import importlib.util
import time
import functools
import json
from typing import TypedDict, Any, List, Optional, Dict, Tuple, Callable, Union, Set
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Default XML tag names (without brackets)
DEFAULT_THINKING_TAG = "think"
DEFAULT_ANSWER_TAG = "answer"

# Execution settings
DEFAULT_EXEC_TIMEOUT = 5.0  # seconds
DEFAULT_TEST_CASES_COUNT = 10
MAX_EXECUTION_ATTEMPTS = 3
SANDBOX_ENABLED = True

# Supported XML tag names for thinking/reasoning
XML_THINKING_TAGS = [
    "think",
    "thinking",
    "thoughts",
    "reasoning",
    "rationale",
]

# Supported XML tag names for answers/solutions
XML_ANSWER_TAGS = [
    "answer",
    "solution",
    "code",
    "result",
    "response",
]

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"


class FormatError(Exception):
    """Error raised when the format of the input is invalid."""
    pass


class ExecutionError(Exception):
    """Error raised when code execution fails."""
    pass


class SecurityError(Exception):
    """Error raised when execution is deemed unsafe."""
    pass


def find_xml_tag_content(text: str, tag_name: str) -> str | None:
    """Find content between XML tags, return None if not found."""
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    
    parts = text.split(start_tag, 1)
    if len(parts) < 2:
        return None
    
    remainder = parts[1]
    parts = remainder.split(end_tag, 1)
    if len(parts) < 2:
        return None
    
    return parts[0].strip()


def extract_thought_solution(
    output: str, 
    thinking_tag: str = DEFAULT_THINKING_TAG,
    answer_tag: str = DEFAULT_ANSWER_TAG,
    require_thinking: bool = False,
    require_answer: bool = True
) -> tuple[str | None, str | None]:
    """
    Extract the thought and solution from the output using specified XML tags.
    If tags aren't found, tries alternative tag names from the supported lists.
    
    Args:
        output: The text containing thoughts and solution
        thinking_tag: The XML tag name for thought section (without brackets)
        answer_tag: The XML tag name for answer section (without brackets)
        require_thinking: Whether to raise error if thought not found
        require_answer: Whether to raise error if answer not found
        
    Returns:
        Tuple of (thought, answer) - either may be None if not required and not found
    """
    # Try with provided tag name
    thought = find_xml_tag_content(output, thinking_tag)
    
    # If thought not found with primary tag, try alternatives
    if thought is None:
        for tag in XML_THINKING_TAGS:
            if tag != thinking_tag:  # Skip the one we already tried
                thought = find_xml_tag_content(output, tag)
                if thought is not None:
                    break
    
    # Try with provided answer tag
    answer = find_xml_tag_content(output, answer_tag)
    
    # If answer not found with primary tag, try alternatives
    if answer is None:
        for tag in XML_ANSWER_TAGS:
            if tag != answer_tag:  # Skip the one we already tried
                answer = find_xml_tag_content(output, tag)
                if answer is not None:
                    break
    
    # If answer still not found, try to use the content after any thinking tag
    if answer is None and require_answer:
        # If we found a thinking section, look for answer after it
        if thought is not None:
            for tag in XML_THINKING_TAGS:
                end_tag = f"</{tag}>"
                if end_tag in output:
                    # Get everything after the end tag
                    parts = output.split(end_tag, 1)
                    if len(parts) > 1 and "<<<<<<< SEARCH" in parts[1]:
                        answer = parts[1].strip()
                        break
    
    # If answer still not found but output contains search/replace blocks, use entire output
    if answer is None and require_answer and "<<<<<<< SEARCH" in output and ">>>>>>> REPLACE" in output:
        answer = output
    
    # Validate based on requirements
    if require_thinking and thought is None:
        raise FormatError("Thinking section not found")
    
    if require_answer and answer is None:
        raise FormatError("Answer/solution section not found")
    
    return thought, answer


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def generate_unified_diff(
    old_code: str,
    new_code: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff."""

    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = dict[str, str]()
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            # Ensure search block can be matched
            # "\n" + search to ensure the indentations are correct
            if len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        new_content_dict[path] = new_content[1:]
    return new_content_dict


def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        if patch:
            patch_dict[path] = patch
    return patch_dict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[ChangeSimilarity]:
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model predicting empty changes to hack the reward.
            # NOTE: this should not happen due to (1) the search == replace check in `apply_code_change`
            # and (2) the `if patch` check in `get_normalized_patch`.
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )
    return similarities


def calculate_change_similarity(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    pred_new_content: dict[str, str],
) -> tuple[float, dict]:
    """
    Compute the similarity between oracle code changes and predicted code changes.
    Note that this function is a general version for measuring change similarity, which can be used
    for code changes in any form, not just search/replace edits. For search/replace edits, use
    `calculate_search_replace_similarity`.

    This is based on the SWE-RL reward calculation logic:
    https://github.com/facebookresearch/swe-rl/blob/main/src/swerl/core/reward.py#L123

    Args:
        code_context: path -> original content of the file. It doesn't need to
            contain the entire codebase, only the files that are affected by the oracle change.
        oracle_new_content: path -> oracle new content of the file after change.
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A float value representing the similarity score (0.0 to 1.0), and a dictionary containing metadata.
    """
    try:
        # Obtain a unified diff for each file, for both the predicted and the oracle changes
        oracle_patch = get_normalized_patch(code_context, oracle_new_content)
        pred_patch = get_normalized_patch(code_context, pred_new_content)
        # Calculate the similarity between the predicted and the oracle patches
        similarities = compute_change_similarities(pred_patch, oracle_patch)
        assert len(similarities) > 0
        similarity_score = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
        return similarity_score, dict(similarities=similarities)
    except FormatError as e:
        return 0.0, dict(error=str(e))


def calculate_search_replace_similarity(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
    thinking_tag: str = DEFAULT_THINKING_TAG,
    answer_tag: str = DEFAULT_ANSWER_TAG,
    require_thinking: bool = False,
) -> tuple[float, dict]:
    """
    The search/replace version of the similarity calculation. It can handle various XML tag formats
    for thought and solution sections, and can even work without explicit answer tags if the content
    contains valid search/replace patterns.

    Args:
        code_context: path -> original content of the file.
        oracle_new_content: path -> oracle new content of the file after change.
        output: The output from the model containing the thought and solution.
        thinking_tag: Optional custom XML tag name for the thinking section
        answer_tag: Optional custom XML tag name for the answer section
        require_thinking: Whether to require the thinking section (default: False)

    Returns:
        A float value representing the similarity score (0.0 to 1.0), and a dictionary containing metadata.
    """
    try:
        # Extract the thought and solution from the output with flexible tag handling
        thought, answer = extract_thought_solution(
            output, 
            thinking_tag=thinking_tag,
            answer_tag=answer_tag,
            require_thinking=require_thinking,
            require_answer=True
        )
        
        # Parse the search/replace edits from the solution
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
            
        # Get the new content of each file after applying the search/replace edits
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        similarity, metadata = calculate_change_similarity(
            code_context, oracle_new_content, pred_new_content
        )
        
        # Only add thought to metadata if it exists
        if thought is not None:
            metadata["thought"] = thought
        metadata["answer"] = answer
        return similarity, metadata
    except FormatError as e:
        return 0.0, dict(error=str(e))


class ASTNodeVisitor(ast.NodeVisitor):
    """
    AST Node visitor that collects node types and their attributes for similarity comparison.
    Ignores position information, variable names, and string literals to focus on structural similarity.
    """
    def __init__(self):
        self.node_types = []
        self.structures = []
        
    def generic_visit(self, node: ast.AST) -> None:
        # Get node type
        node_type = type(node).__name__
        self.node_types.append(node_type)
        
        # For certain node types, collect structural information
        # but ignore variable names, string literals, and positions
        if isinstance(node, ast.FunctionDef):
            # Track function structure (args count, has default, has return, etc)
            self.structures.append(f"Function:{len(node.args.args)}:{bool(node.returns)}")
        elif isinstance(node, ast.ClassDef):
            # Track class inheritance structure
            self.structures.append(f"Class:{len(node.bases)}")
        elif isinstance(node, ast.Call):
            # Track call structure (arg count)
            self.structures.append(f"Call:{len(node.args)}:{len(node.keywords)}")
        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            # Track loop presence
            self.structures.append(f"Loop:{node_type}")
        elif isinstance(node, ast.If) or isinstance(node, ast.IfExp):
            # Track conditional structure
            self.structures.append("Conditional")
        
        # Continue to visit child nodes
        super().generic_visit(node)


def parse_ast_safely(code: str) -> Optional[ast.AST]:
    """Parse code into AST, handling syntax errors gracefully."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def get_ast_features(code: str) -> tuple[List[str], List[str]]:
    """Extract AST features from code."""
    tree = parse_ast_safely(code)
    if tree is None:
        return [], []
        
    visitor = ASTNodeVisitor()
    visitor.visit(tree)
    return visitor.node_types, visitor.structures


def calculate_ast_similarity(
    node_types1: List[str], 
    node_types2: List[str],
    structures1: List[str], 
    structures2: List[str]
) -> float:
    """Calculate similarity between two sets of AST features."""
    # Calculate node type similarity
    type_similarity = difflib.SequenceMatcher(
        None, node_types1, node_types2, autojunk=False
    ).ratio()
    
    # Calculate structural similarity
    if structures1 and structures2:
        structure_similarity = difflib.SequenceMatcher(
            None, structures1, structures2, autojunk=False
        ).ratio()
    else:
        structure_similarity = 0.0
    
    # Weight node types more (0.6) than structures (0.4)
    return (type_similarity * 0.6) + (structure_similarity * 0.4)




def is_safe_to_execute(code: str) -> bool:
    """
    Check if code is safe to execute by looking for dangerous patterns.
    
    This is a basic check and not foolproof. It looks for common patterns
    that could indicate malicious code.
    """
    if not SANDBOX_ENABLED:
        return True
        
    # Pattern list of potentially dangerous operations
    dangerous_patterns = [
        # System operations
        r"os\s*\.\s*system",
        r"subprocess",
        r"exec\s*\(",
        r"eval\s*\(",
        r"execfile",
        r"compile\s*\(",
        # File operations
        r"open\s*\(",
        r"file\s*\(",
        r"os\s*\.\s*remove",
        r"os\s*\.\s*unlink",
        r"shutil\s*\.\s*rmtree",
        # Network operations
        r"socket\s*\.",
        r"urllib",
        r"requests",
        r"http",
        # Module loading
        r"__import__",
        r"importlib",
        r"imp\s*\.",
        # Code injection
        r"globals\s*\(",
        r"locals\s*\(",
        r"getattr\s*\(",
        r"setattr\s*\(",
        # System info
        r"sys\s*\.",
        # Other dangerous modules
        r"pickle",
        r"shelve",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False
            
    return True


def extract_function_signature(code: str) -> Tuple[str, List[str], bool]:
    """Extract function name, parameters, and whether it has a return value."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function name
                function_name = node.name
                
                # Get parameter names
                param_names = []
                for arg in node.args.args:
                    param_names.append(arg.arg)
                    
                # Check if function has a return statement
                has_return = False
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Return) and subnode.value is not None:
                        has_return = True
                        break
                        
                return function_name, param_names, has_return
    except Exception:
        pass
        
    return "", [], False


def generate_test_cases(function_name: str, parameters: List[str], count: int = DEFAULT_TEST_CASES_COUNT) -> List[Dict]:
    """
    Generate random test cases for a function based on its signature.
    
    Args:
        function_name: Name of the function
        parameters: List of parameter names
        count: Number of test cases to generate
        
    Returns:
        List of test cases, each with input values and expected output set to None
    """
    test_cases = []
    
    for _ in range(count):
        inputs = {}
        for param in parameters:
            # Generate a random input based on parameter name hints
            if "id" in param.lower() or "count" in param.lower() or "num" in param.lower():
                # Likely an integer
                inputs[param] = random.randint(1, 100)
            elif "name" in param.lower() or "str" in param.lower() or "text" in param.lower():
                # Likely a string
                options = ["apple", "banana", "orange", "hello", "world", "python", "code"]
                inputs[param] = random.choice(options)
            elif "list" in param.lower() or "items" in param.lower() or "elements" in param.lower():
                # Likely a list
                list_length = random.randint(3, 10)
                inputs[param] = [random.randint(1, 100) for _ in range(list_length)]
            elif "dict" in param.lower() or "map" in param.lower():
                # Likely a dictionary
                dict_size = random.randint(2, 5)
                inputs[param] = {f"key{i}": random.randint(1, 100) for i in range(dict_size)}
            elif "bool" in param.lower() or "flag" in param.lower():
                # Likely a boolean
                inputs[param] = random.choice([True, False])
            elif "float" in param.lower() or "price" in param.lower() or "rate" in param.lower():
                # Likely a float
                inputs[param] = round(random.uniform(0.1, 100.0), 2)
            else:
                # Default to an integer if we can't determine the type
                inputs[param] = random.randint(1, 100)
                
        test_case = {
            "inputs": inputs,
            "expected_output": None  # We don't know the expected output yet
        }
        test_cases.append(test_case)
        
    return test_cases


def execute_code_safely(code: str, func_name: str, inputs: Dict[str, Any], timeout: float = DEFAULT_EXEC_TIMEOUT) -> Any:
    """
    Execute a function from the given code with the given inputs in a safe manner.
    
    Args:
        code: Python code containing the function to execute
        func_name: Name of the function to call
        inputs: Dictionary of parameter name -> value to pass to the function
        timeout: Maximum execution time in seconds
        
    Returns:
        The return value of the function
        
    Raises:
        SecurityError: If the code is deemed unsafe
        ExecutionError: If execution fails
        TimeoutError: If execution takes too long
    """
    if not is_safe_to_execute(code):
        raise SecurityError("Code contains potentially unsafe operations")
    
    # Create a temporary module
    module_name = f"temp_module_{random.randint(10000, 99999)}"
    
    # Create a safe globals dictionary with limited built-ins
    safe_globals = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if name not in [
                "open", "exec", "eval", "compile", "execfile", "__import__",
                "globals", "locals", "input", "memoryview", "breakpoint"
            ]
        }
    }
    
    try:
        # Compile the code
        compiled_code = compile(code, "<string>", "exec")
        
        # Execute the code in the safe globals
        exec(compiled_code, safe_globals)
        
        # Get the function
        if func_name not in safe_globals:
            raise ExecutionError(f"Function '{func_name}' not found in the code")
        
        func = safe_globals[func_name]
        
        # Execute the function with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **inputs)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                raise ExecutionError(f"Execution timed out after {timeout} seconds")
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        trace = traceback.format_exc()
        raise ExecutionError(f"{error_type}: {error_msg}\n{trace}")


def calculate_execution_similarity(
    ground_truth_code: str, 
    predicted_code: str, 
    test_cases: List[Dict] = None,
    timeout: float = DEFAULT_EXEC_TIMEOUT,
    max_attempts: int = MAX_EXECUTION_ATTEMPTS
) -> Tuple[float, Dict]:
    """
    Calculate similarity based on execution results for the same inputs.
    
    Args:
        ground_truth_code: The reference/ground truth code
        predicted_code: The generated/predicted code
        test_cases: Optional list of test cases to use. If None, they will be generated
        timeout: Maximum execution time per test case in seconds
        max_attempts: Maximum number of attempts to execute each function
        
    Returns:
        A tuple containing (similarity_score, metadata) where similarity_score 
        is between 0.0 and 1.0, and metadata contains execution details
    """
    # Extract function signature from ground truth code
    gt_func_name, gt_params, gt_has_return = extract_function_signature(ground_truth_code)
    pred_func_name, pred_params, pred_has_return = extract_function_signature(predicted_code)
    
    # Initialize metadata
    metadata = {
        "ground_truth_function": gt_func_name,
        "predicted_function": pred_func_name,
        "ground_truth_params": gt_params,
        "predicted_params": pred_params,
        "execution_valid": False,
        "test_results": [],
        "success_rate": 0.0,
    }
    
    # Check if we could extract functions
    if not gt_func_name or not pred_func_name or not gt_has_return or not pred_has_return:
        return 0.0, metadata
    
    # Handle case where parameter names differ but should match in order
    if len(gt_params) == len(pred_params) and gt_params != pred_params:
        # Create a parameter mapping from ground truth to predicted
        param_mapping = {gt_param: pred_param for gt_param, pred_param in zip(gt_params, pred_params)}
    else:
        param_mapping = None

    # If no test cases provided, generate them
    if test_cases is None:
        test_cases = generate_test_cases(gt_func_name, gt_params)
    
    # Initialize counters
    successful_tests = 0
    total_tests = len(test_cases)
    
    # Run each test case
    for test_case in test_cases:
        gt_inputs = test_case["inputs"]
        
        # Map inputs if parameter names differ
        if param_mapping:
            pred_inputs = {param_mapping[k]: v for k, v in gt_inputs.items()}
        else:
            pred_inputs = gt_inputs.copy()
        
        test_result = {
            "inputs": gt_inputs,
            "ground_truth_success": False,
            "predicted_success": False,
            "outputs_match": False,
        }
        
        # Execute ground truth code (multiple attempts in case of transient errors)
        gt_output = None
        gt_error = None
        
        for attempt in range(max_attempts):
            try:
                gt_output = execute_code_safely(ground_truth_code, gt_func_name, gt_inputs, timeout)
                test_result["ground_truth_success"] = True
                test_result["ground_truth_output"] = gt_output
                break
            except (ExecutionError, SecurityError) as e:
                gt_error = str(e)
                time.sleep(0.1)  # Small delay before retry
        
        if not test_result["ground_truth_success"]:
            test_result["ground_truth_error"] = gt_error
            metadata["test_results"].append(test_result)
            continue
        
        # Execute predicted code
        pred_output = None
        pred_error = None
        
        for attempt in range(max_attempts):
            try:
                pred_output = execute_code_safely(predicted_code, pred_func_name, pred_inputs, timeout)
                test_result["predicted_success"] = True
                test_result["predicted_output"] = pred_output
                break
            except (ExecutionError, SecurityError) as e:
                pred_error = str(e)
                time.sleep(0.1)  # Small delay before retry
        
        if not test_result["predicted_success"]:
            test_result["predicted_error"] = pred_error
            metadata["test_results"].append(test_result)
            continue
        
        # Compare outputs
        try:
            outputs_match = False
            
            # Handle different types
            if type(gt_output) != type(pred_output):
                # Try string conversion for simple types
                outputs_match = str(gt_output) == str(pred_output)
            elif isinstance(gt_output, (list, tuple)):
                # For lists, check if they have the same elements
                if len(gt_output) == len(pred_output):
                    try:
                        outputs_match = all(
                            str(gt_item) == str(pred_item) 
                            for gt_item, pred_item in zip(gt_output, pred_output)
                        )
                    except Exception:
                        outputs_match = False
            else:
                # Direct comparison for other types
                try:
                    outputs_match = gt_output == pred_output
                except Exception:
                    outputs_match = False
            
            test_result["outputs_match"] = outputs_match
            
            if outputs_match:
                successful_tests += 1
                
        except Exception as e:
            test_result["comparison_error"] = str(e)
        
        metadata["test_results"].append(test_result)
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = successful_tests / total_tests
    else:
        success_rate = 0.0
    
    metadata["success_rate"] = success_rate
    metadata["execution_valid"] = True
    
    return success_rate, metadata


def calculate_code_similarity(
    ground_truth_code: str,
    predicted_code: str,
    use_ast: bool = True,
    use_execution: bool = False,
    test_cases: List[Dict] = None,
    execution_timeout: float = DEFAULT_EXEC_TIMEOUT,
) -> tuple[float, dict]:
    """
    Calculate the similarity between ground truth and predicted code.
    
    Args:
        ground_truth_code: The reference/expected/ground truth code
        predicted_code: The generated/predicted code
        use_ast: Whether to include Python AST-based similarity
        use_execution: Whether to include execution-based similarity
        test_cases: Optional test cases for execution-based similarity
        execution_timeout: Timeout for execution-based similarity
        
    Returns:
        A tuple containing (similarity_score, metadata)
        where similarity_score is between 0.0 and 1.0, and
        metadata contains additional information about the comparison
    """
    # Clean up whitespace to avoid trivial differences
    ground_truth_code = ground_truth_code.strip()
    predicted_code = predicted_code.strip()
    
    # Calculate text-based similarity (edit distance)
    text_similarity = difflib.SequenceMatcher(
        None, 
        ground_truth_code, 
        predicted_code,
        autojunk=False
    ).ratio()
    
    # Generate a unified diff for visual inspection
    diff = generate_unified_diff(ground_truth_code, predicted_code)
    
    # Initialize metadata
    metadata = {
        "text_similarity": text_similarity,
        "diff": diff,
        "ground_truth_length": len(ground_truth_code),
        "predicted_length": len(predicted_code),
    }
    
    # Calculate weights based on which similarity measures are used
    weight_text = 1.0
    weight_ast = 0.0
    weight_execution = 0.0
    valid_measures = 1  # Text similarity is always valid
    
    # Calculate AST-based similarity if requested and code is valid Python
    ast_similarity = 0.0
    if use_ast:
        try:
            # Extract AST features
            gt_node_types, gt_structures = get_ast_features(ground_truth_code)
            pred_node_types, pred_structures = get_ast_features(predicted_code)
            
            # Calculate AST similarity if both are valid Python
            if gt_node_types and pred_node_types:
                ast_similarity = calculate_ast_similarity(
                    gt_node_types, pred_node_types,
                    gt_structures, pred_structures
                )
                metadata["ast_similarity"] = ast_similarity
                metadata["ast_valid"] = True
                weight_ast = 1.0
                valid_measures += 1
            else:
                metadata["ast_valid"] = False
        except Exception:
            # If AST parsing fails for any reason, fall back to text similarity
            metadata["ast_valid"] = False
        
    # Calculate execution-based similarity if requested
    execution_similarity = 0.0
    if use_execution:
        try:
            exec_sim, exec_meta = calculate_execution_similarity(
                ground_truth_code, 
                predicted_code,
                test_cases=test_cases,
                timeout=execution_timeout
            )
            execution_similarity = exec_sim
            metadata["execution_similarity"] = exec_sim
            metadata.update({f"execution_{k}": v for k, v in exec_meta.items()})
            
            if exec_meta.get("execution_valid", False):
                weight_execution = 1.0
                valid_measures += 1
            else:
                metadata["execution_valid"] = False
        except Exception as e:
            metadata["execution_valid"] = False
            metadata["execution_error"] = str(e)
    
    # Normalize weights
    total_weight = weight_text + weight_ast + weight_execution
    if total_weight > 0:
        weight_text /= total_weight
        weight_ast /= total_weight
        weight_execution /= total_weight
    
    # Calculate final similarity as weighted average of available measures
    similarity = (text_similarity * weight_text + 
                 ast_similarity * weight_ast + 
                 execution_similarity * weight_execution)
    
    metadata["similarity"] = similarity
    metadata["weights"] = {
        "text": weight_text,
        "ast": weight_ast,
        "execution": weight_execution
    }
    
    return similarity, metadata
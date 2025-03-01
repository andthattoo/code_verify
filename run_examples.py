#!/usr/bin/env python3
"""
Run the code_verify examples.
"""

from code_verify.examples import (
    example_basic_usage,
    example_multiple_changes,
    example_flexible_tags,
    example_code_similarity,
    example_ast_similarity,
    example_change_similarity,
    example_execution_similarity
)

def main():
    print("=" * 40)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 40)
    example_basic_usage()
    
    print("\n" + "=" * 40)
    print("EXAMPLE 2: Multiple Changes")
    print("=" * 40)
    example_multiple_changes()

    print("\n" + "=" * 40)
    print("EXAMPLE 3: Flexible Tag Handling")
    print("=" * 40)
    example_flexible_tags()

    print("\n" + "=" * 40)
    print("EXAMPLE 4: Code Similarity Measurement")
    print("=" * 40)
    example_code_similarity()

    print("\n" + "=" * 40)
    print("EXAMPLE 5: AST-Based Similarity")
    print("=" * 40)
    example_ast_similarity()

    print("\n" + "=" * 40)
    print("EXAMPLE 6: Code Change Similarity")
    print("=" * 40)
    example_change_similarity()
    
    print("\n" + "=" * 40)
    print("EXAMPLE 7: Execution-Based Similarity")
    print("=" * 40)
    example_execution_similarity()

if __name__ == "__main__":
    main()
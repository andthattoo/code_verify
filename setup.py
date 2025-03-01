from setuptools import setup, find_packages

setup(
    name="code_verify",
    version="0.1.0",
    packages=find_packages(),
    description="A library for verifying code changes and calculating similarity between patches",
    author="andthattoo",
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "tsed": ["apted", "tree_sitter_languages"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
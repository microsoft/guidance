import sys

# Check if 'setup.py' is run directly with 'build'
# TODO: Consider generalizing this check further?
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        raise SystemExit(
            "Error: Direct invocation of 'setup.py build' is not recommended."
            "Please use 'pip' to build and install this package, like so:\n"
            "  pip install . (for the current directory)\n"
            "  pip install -e . (for an editable install)\n"
            "  pip wheel . --no-deps (to build a wheel)"
        )

import os
import re
import codecs
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

here = os.path.abspath(os.path.dirname(__file__))

llamacpp_requires = ["llama-cpp-python==0.3.7"]
transformers_requires = ["transformers==4.48.2"]

install_requires = [
    "diskcache",
    "numpy",
    "ordered_set",
    "platformdirs",
    "pydantic",
    "referencing",
    "requests",
    "psutil",
    "tiktoken>=0.3",
    "guidance-stitch",
    "llguidance==0.6.12",
]

# Our basic list of 'extras'
extras_requires = {
    "azureai": ["openai>=1.0"],
    "openai": ["openai>=1.0"],
    "server": ["fastapi-slim", "uvicorn"],
}

# Create the union of all our requirements
all_requires = set()
for v in extras_requires.values():
    all_requires = all_requires.union(v)

# Required for builds etc.
doc_requires = [
    "ipython",
    "nbsphinx",
    "numpydoc",
    "sphinx_rtd_theme",
    "sphinx",
    "ipykernel",
    "huggingface_hub",
    "llama-cpp-python",
]
unittest_requires = [
    "anytree",
    "jsonschema",
    "pytest",
    "pytest-cov",
    "tokenizers",
]
test_requires = [
    "types-regex",
    "types-requests",
    "types-jsonschema",
    "requests",
    "azure-identity",
    "bitsandbytes",
    "jupyter",
    "papermill",
    "protobuf",
    "sentencepiece",
    "torch",
    "transformers",
    "mypy==1.9.0",
] + unittest_requires

bench_requires = [
    "pandas",
    "huggingface_hub",
    "langchain_benchmarks",
    "langchain-community",
    "langsmith",
    "json_stream",
    "llama-cpp-python",
    "setuptools",
    "powerlift"
]


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="guidance",
    version=find_version("guidance", "__init__.py"),
    url="https://github.com/guidance-ai/guidance",
    author="Guidance Maintainers",
    author_email="maintainers@guidance-ai.org",
    description="A guidance language for controlling large language models.",
    long_description="Guidance enables you to control modern language models more effectively and efficiently than traditional prompting or chaining. Guidance programs allow you to interleave generation, prompting, and logical control into a single continuous flow matching how the language model actually processes the text.",
    packages=find_packages(exclude=["notebooks", "client", "tests", "tests.*"]),
    package_data={"guidance": ["resources/*"]},
    ext_modules=[
        Pybind11Extension(
            "guidance.cpp", ["guidance/_cpp/main.cpp", "guidance/_cpp/byte_trie.cpp"]
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "all": all_requires,
        "unittest": unittest_requires,
        "llamacpp": llamacpp_requires,
        "transformers": transformers_requires,
        "test": test_requires,
        "docs": doc_requires,
        "bench": bench_requires,
        **extras_requires,
    },
)

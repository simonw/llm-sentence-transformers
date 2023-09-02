# llm-embed-sentence-transformers

[![PyPI](https://img.shields.io/pypi/v/llm-embed-sentence-transformers.svg)](https://pypi.org/project/llm-embed-sentence-transformers/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-embed-sentence-transformers?include_prereleases&label=changelog)](https://github.com/simonw/llm-embed-sentence-transformers/releases)
[![Tests](https://github.com/simonw/llm-embed-sentence-transformers/workflows/Test/badge.svg)](https://github.com/simonw/llm-embed-sentence-transformers/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-embed-sentence-transformers/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for embedding models using [sentence-transformers](https://www.sbert.net/)

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-embed-sentence-transformers
```
## Configuration

TODO

## Usage

TODO

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-embed-sentence-transformers
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```

# DUA-VLM-POC
DUA-VLM-PLC is a proof-of-concept testing the feasibility of Vision Language Models (VLMs) to classify and produced structured data from documents and images in support of a [Disaster Unemployment Assistance](https://www.disasterassistance.gov/get-assistance/forms-of-assistance/4466) (DUA) claim. 

This proof-of-concept relies on [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) for inference and fine-tunrning of VLMs on Apple Silicon devices.

This proof-of-concept has been tested with the following models:

* mlx-community/SmolVLM-Instruct-bf16
* mlx-community/llava-interleave-qwen-7b-4bit
* mlx-community/Phi-3.5-vision-instruct-bf16
---

## Table of Contents


- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
    - [Tests](#tests)
    - [Models](#models)
    - [Output and examples](#output-and-examples)
- [To-do](#to-do)
- [Changelog](#changelog)

---

## Requirements 

* Apple Silicon. This proof-of-concept makes use of Apple's MLX to run models on Apple Silicon. It will not work, and has not been tested, on non-Apple Silicon. 
* Minimum 32GB RAM. DUA-VLM-POC has been developed and tested on M1 Pro with 32GB of RAM. 

---

## Installation

You will need [uv](https://docs.astral.sh/uv/) to run this proof-of-concept.

---

## Usage

DUA-VLM-POC is designed to run arbitrary tests against arbitrary vision language models.

Run dua-vlm-poc by using uv: 

`uv run dua-vlm-poc.py`

This will download and install any required Python packages and models. 

By default, running dua-vlm-poc.py will:

* use the models specified in models.txt
* run the tests specified in tests.csv
* output the results in results.csv

You can override these defaults by specifing arguments:

`uv run dua-vlm-poc.py --models my_models.txt --tests my_tests.csv --output my_results.csv`

will use the specified files as sources for the model list, tests, and output file.


### Models

Models are specified in models.txt. Models are referenced by their huggingface path, e.g. mlx-community/SmolLM3-3B-Base-bf16.

models.txt is expected to be in the current working directory of the script.

You must use MLX model weights that run on Apple Silicon. As of writing, Hugging Face hosts over 2,700 [MLX models](https://huggingface.co/mlx-community/models).

Example models.txt

```
mlx-community/SmolVLM-Instruct-bf16
mlx-community/llava-interleave-qwen-7b-4bit
mlx-community/Phi-3.5-vision-instruct-bf16
```

### Tests

Tests are provided in tests.csv in this format:

|Column name|Description|Example
---|---|---
test_description|Quoted string describing the test|"Extract the proprietor name from a provided form"
prompt|Quoted string with the prompt text|"What is the name of the proprietor?"|
image|Quoted string containing the relative path to the image file for the test|"images/test-002.jpg"
expected_result|Quoted string containing an expected result|"Ricky Nelson"

DUA-VLM-POC implements a naive test:

* the prompt output is transformed to lowercase
* the test string is transformed to lowercase
* if the test string is in the prompt response, return `True`

### Acceptable image formats
DUA-VLM-POC uses [Pillow](https://python-pillow.github.io) to process images and has been tested with png and jpeg formats.


## Output and examples

Output is appended to results.csv in the following format:


### Passed test example

|Column name|Description|Example
---|---|---
timestamp|Timezone timestamp of when the prompt was run|2025-07-29T16:25:51
model|The Hugging Face model path used for the test|mlx-community/SmolVLM-Instruct-bf16
test_description|The provided test description from the test file|Extract the proprietor name from a provided form
prompt|The provided prompt from the test file|What is the name of the proprietor?
output|The output response from the model|The name of the proprietor is Ricky Nelson.
expected_result|The expected result string from the test file|Ricky Nelson
check|"True" if the test `expected_result` string is present in the output string, otherwise `False`|True


### Failed test example

|Column name|Description|Example
---|---|---
timestamp|Timezone timestamp of when the prompt was run|2025-07-29T16:26:12
model|The Hugging Face model path used for the test|mlx-community/llava-interleave-qwen-7b-4bit
test_description|The provided test description from the test file|Extract the proprietor name from a provided form
prompt|The provided prompt from the test file|What is the name of the proprietor?
output|The output response from the model|The name of the proprietor is not provided in the image.
expected_result|The expected result string from the test file|Ricky Nelson
check|"True" if the test `expected_result` string is present in the output string, otherwise `False`|False

----

## To-do
* Do logging properly
* Tidy up the console output
* Stalls on requiring passing the argument `trust_remote_code=True` for some huggingface models requiring y/n from user
* Implement args to allow alternatives to results.csv, tests.csv, and models.txt for users to specify their own results, tests, and models
* ~~move from personal repo to VLG repo~~
* ~~test against the expected result, duh~~
* ~~the csv should include the test image too, duh~~
* ~~tests should be a tuple: prompt, test data, expected result~~

---

# Changelog

## 0.04 
* lines in models.txt will be ignored if they start with a comment (e.g. "# dont-use-this-model/modelname")
* lines in tests.csv will be ignored if they start with a comment (#)
* a models file, output file, and tests file can be specified as arguments, otherwise the defaults are models.txt, results.csv, and tests.csv in the working directory
* Tidied up readme.md
    * Added Requirements
    * Added Installation
    * Added Usage
    * Added model documentation
    * Added test documentation
    * Added output examples


## 0.03
* Only load model once
* Actually do the test

## 0.02 
Moved test/prompt/image/expected result into tests.csv. Here's the columns:
"test_description","prompt","image","expected_result"

Not actually using the eval string yet.

## 0.01 
Prompt is hardcoded in prompts.txt
Models are hardcoded in models.txt
No evals yet.

Only iterates over models.
No support for multiple prompts.

Barely works tbh.



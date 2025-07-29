# Prototyping code for DUA VLM tests

* Use [uv](https://docs.astral.sh/uv/guides/install-python/) to run this.
* Apple Silicon using MLX/[MXL-VLM](https://github.com/Blaizzy/mlx-vlm), no CUDA etc

# Where stuff goes
* Put your huggingface models in models.txt
* Put your prompt in prompts.txt
* Put your test images in images/
* Results are saved in results.csv with a timestamp, the model used, the prompt used, and then the text result

# Changelog

## 0.1 
Prompt is hardcoded in prompts.txt
Models are hardcoded in models.txt
No evals yet.

Only iterates over models.
No support for multiple prompts.

Barely works tbh.

## 0.2 

Moved test/prompt/image/expected result into tests.csv. Here's the columns:
"test_description","prompt","image","expected_result"

Not actually using the eval string yet.

# todo

* the csv should include the test image too, duh
* tests should be a tuple: prompt, test data, expected result


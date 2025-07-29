import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pathlib import Path
from datetime import datetime

 
import csv

MODELS_FILE = "models.txt"
OUTPUT_FILE = "results.csv"
TEST_PROMPT = "Provide the form number, year,  proprietor name,  principal crop or activity, and net profit or loss as json"
TEST_IMAGE = "images/test-002.jpg"
IMG_PATH = Path(TEST_IMAGE)
TEST_FILE = "tests.csv"

def load_tests(test_file):
    tests = []

    with open(test_file, newline='', encoding='utf-8') as test_file:
        csv_reader = csv.DictReader(test_file)
        for row in csv_reader:
            tests.append(dict(row))
    return tests


def load_models_list(models_file):
    """Load the models from the models file"""
    with open(models_file, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def load_model(model_path):
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config

def run_prompt(model, processor, config, prompt, image):
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
    response = generate(model, processor, formatted_prompt, image, verbose=False)
    return response

def run_prompt_on_model(model_path, prompt, image):
    model, processor = load(model_path)
    config = load_config(model_path)
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
    response = generate(model, processor, formatted_prompt, image, verbose=False)
    return response

def check_result(prompt_output, test):
    return test.lower() in prompt_output.lower()


def main():
    models = load_models_list(MODELS_FILE)
    results = []
    tests = load_tests(TEST_FILE)


    for model_path in models:
        timestamp = datetime.now().isoformat(timespec="seconds")

        model, processor, config = load_model(model_path)

        for test in tests:
            print(f"Running test {test['test_description']} on {model_path}")

            prompt = test['prompt']
            image = [Image.open(test['image'])]

            try:
                # output = run_prompt_on_model(model_path, prompt, image)
                output = run_prompt(model, processor, config, prompt, image)
                results.append({"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": check_result(output.text, test['expected_result'])})
                print(output)
            except Exception as e:
                results.append({"model": model_path, "output": f"Error {e}"})


    # write to CSV
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "model", "test_description", "prompt", "output", "expected_result", "check"])
        # writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pillow_heif import register_heif_opener
from pathlib import Path
from datetime import datetime
import csv
import argparse
import json
import logging
from os import listdir
from os.path import isfile, join

DEFAULT_MODELS_FILE = "models.txt"
DEFAULT_OUTPUT_FILE = "results.csv"
DEFAULT_TEST_FILE = "tests.json"

def load_tests_json(tests_file):
    print(f"Trying to load json tests file {tests_file}")
    with open(tests_file, encoding='utf-8') as test_file:
        tests_json = json.load(test_file)
    return tests_json

def load_test_images(test_image_path):
    '''Returns an array of the images in a provided image path'''
    try:
        test_images = [f for f in listdir(test_image_path) if isfile(join(test_image_path, f))]
    except Exception as e:
        print(f"Oops: {e}")
    return test_images

def run_tests_json(models_file, tests_file, output_file):

    models = load_models_list(models_file)
    tests = load_tests_json(tests_file)

    # print(tests)

    # Initialize where we're going to be storing the results of all tests
    results = []

    # iterate over the models we're going to test
    for model_path in models:
        timestamp = datetime.now().isoformat(timespec="seconds")
        model, processor, config = load_model(model_path)

        # iterate over the tests we're going to test
        # each test is one test, prompt, image_directory, expected result
        # iterate over the tests by running the prompt against each image in the image directory
        for test in tests:
            # next, iterate over the prompts we're going to test

            # we will need to run each prompt over each image
            # next, iterate over the images we need to test against

            # get all the images
            test_images = load_test_images(test['image_directory'])

            print(f"Running test {test['test_description']} on {model_path} using {test['image_directory']} and {test_images}")
            print(f"found results to check against: {test['expected_result']}")
            for image in test_images:

                image_path = join(test['image_directory'], image)
                
                prompt = test['prompt']
                print(f"running prompt {test['prompt']} on image {image_path}")
                register_heif_opener()
                image = [Image.open(image_path)]
                this_result = []

                try:
                    # run the prompt against the image
                    output = run_prompt(model, processor, config, prompt, image)

                    # this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": check_result(output.text, test['expected_result'])}]
                    
                    
                    results.append(this_result)
                    # print(output)

                except Exception as e:
                    this_result = [{"model": model_path, "output": f"Error {e}"}]
                    results.append(this_result)

                # in this approach, we want one row per test, and we return True if any of the expected results are in the prompt output
                # so we just take output.text and the list of expected_results

                check_strings = test['expected_result']

                test_satisfied = False

                for check in check_strings:
                    if str(check).lower() in output.text.lower():
                        test_satisfied = True
                        break
                
                this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": test_satisfied}]
                results.append(this_result)                
                write_results(output_file, this_result)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLX-VLM tests on specified models.")
    parser.add_argument("--models", default=DEFAULT_MODELS_FILE, help="Provide the path to model file listing huggingface models (default: models.txt)")
    parser.add_argument("--tests", default=DEFAULT_TEST_FILE, help="Provide the path to the csv file containing tests (default: tests.csv)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help = "Provide the path to the output csv file (default: results.csv")
    return parser.parse_args()
    

def load_tests(test_file):
    """Takes a path to a tests.txt file and returns a list of dicts of tests"""
    tests = []

    with open(test_file, newline='', encoding='utf-8') as test_file:
        csv_reader = csv.DictReader(test_file)
        for row in csv_reader:
            # Skip a row if the text_description column or first row is commented out, e.g. starts with "#""
            if row["test_description"].strip().startswith("#"):
                    continue
            tests.append(dict(row))
    return tests


def load_models_list(models_file):
    """Takes a path to a models.txt file and returns a list of huggingface models """
    with open(models_file, "r") as f:
        return [
            line.strip()
            for line in f 
            if line.strip() and not line.strip().startswith("#")
            ]
    
def load_model(model_path):
    """Takes a huggingface model path and returns an mlx-vlm model, processor, and prompt configuration to use"""
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config

def run_prompt(model, processor, config, prompt, image):
    """Runs a provided prompt and returns the mlx-vlm generator response"""
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
    response = generate(model, processor, formatted_prompt, image, verbose=False)
    return response

def check_result(prompt_output, test):
    # actually what we want to do here is get the prompt output, and the list of strings to check against.
    # if *any* of the strings to check against are in the prompt output, then we returnt true

    """Provide the prompt output and a test string. Normalizes / lowercases the strings and returns whether the test string is present in the prompt output."""
    if test:
        return test.lower() in prompt_output.lower()


def write_results(file_path, results):
    file_exists = Path(file_path).exists()

    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "model", "test_description", "prompt", "output", "expected_result", "check"])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)

def main():

    args = parse_args()

    models_file = args.models
    output_file = args.output
    tests_file = args.tests

    models = load_models_list(models_file)
    tests = load_tests_json(tests_file)

    # print(tests)

    # Initialize where we're going to be storing the results of all tests
    results = []

    # iterate over the models we're going to test
    for model_path in models:
        timestamp = datetime.now().isoformat(timespec="seconds")
        model, processor, config = load_model(model_path)

        # iterate over the tests we're going to test
        # each test is one test, prompt, image_directory, expected result
        # iterate over the tests by running the prompt against each image in the image directory
        for test in tests:
            # next, iterate over the prompts we're going to test

            # we will need to run each prompt over each image
            # next, iterate over the images we need to test against

            # get all the images
            test_images = load_test_images(test['image_directory'])

            print(f"Running test {test['test_description']} on {model_path} using {test['image_directory']} and {test_images}")
            print(f"found results to check against: {test['expected_result']}")
            for image in test_images:

                image_path = join(test['image_directory'], image)
                
                prompt = test['prompt']
                print(f"running prompt {test['prompt']} on image {image_path}")
                register_heif_opener()
                image = [Image.open(image_path)]
                this_result = []

                try:
                    # run the prompt against the image
                    output = run_prompt(model, processor, config, prompt, image)

                    # this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": check_result(output.text, test['expected_result'])}]
                    
                    
                    results.append(this_result)
                    # print(output)

                except Exception as e:
                    this_result = [{"model": model_path, "output": f"Error {e}"}]
                    results.append(this_result)

                # in this approach, we want one row per test, and we return True if any of the expected results are in the prompt output
                # so we just take output.text and the list of expected_results

                check_strings = test['expected_result']

                test_satisfied = False

                for check in check_strings:
                    if str(check).lower() in output.text.lower():
                        test_satisfied = True
                        break
                
                this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": test_satisfied}]
                results.append(this_result)                
                write_results(output_file, this_result)

def old_main():

    args = parse_args()

    models_file = args.models
    output_file = args.output
    tests_file = args.tests
    
    models = load_models_list(models_file)
    results = []

    # this is the csv version of tests
    tests = load_tests(tests_file)

    # print(f"Using models file: {models_file}")
    # print(f"Using tests file: {tests_file}")
    # print(f"Using output file: {output_file}")

    for model_path in models:
        timestamp = datetime.now().isoformat(timespec="seconds")

        model, processor, config = load_model(model_path)

        for test in tests:
            print(f"Running test {test['test_description']} on {model_path}")

            prompt = test['prompt']
            register_heif_opener()
            image = [Image.open(test['image'])]
            this_result = []

            try:
                output = run_prompt(model, processor, config, prompt, image)
                this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "prompt": prompt, "output": output.text, "expected_result": test['expected_result'], "check": check_result(output.text, test['expected_result'])}]
                results.append(this_result)

                print(output)
            except Exception as e:
                this_result = [{"model": model_path, "output": f"Error {e}"}]
                results.append(this_result)

            write_results(output_file, this_result)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

    # testing json version

    # args = parse_args()

    # models_file = args.models
    # output_file = args.output
    # tests_file = args.tests
    
    # print(f"Using models file: {models_file}")
    # print(f"Using tests file: {tests_file}")
    # print(f"Using output file: {output_file}")

    # run_tests_json(models_file, tests_file, output_file)
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
DEFAULT_LOGFILE = "logs/DUA-VLM-POC.log"
DEFAULT_SYSTEM_PROMPT_FILE = "system_prompt.json"

def load_system_prompt(system_prompt_file):
    with open(system_prompt_file, encoding='utf-8') as f:
        system_prompt_json = json.load(f)
        system_prompt = system_prompt_json[0]['system_prompt']
    return system_prompt

def load_tests_json(tests_file):
    # print(f"Trying to load json tests file {tests_file}")
    with open(tests_file, encoding='utf-8') as test_file:
        tests_json = json.load(test_file)
    
    for idx, test in enumerate(tests_json, start=1): 
        logger.debug('Test %s of %s: %s', idx, len(tests_json), test['test_description'])    
        
    return tests_json

def load_test_images(test_image_path):
    '''Returns an array of the files in a path'''
    try:
        test_images = [f for f in listdir(test_image_path) if isfile(join(test_image_path, f))]
    except Exception as e:
        print(f"Oops: {e}")
    return test_images


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLX-VLM tests on specified models.")
    parser.add_argument("--models", default=DEFAULT_MODELS_FILE, help="Provide the path to model file listing huggingface models (default: models.txt)")
    parser.add_argument("--tests", default=DEFAULT_TEST_FILE, help="Provide the path to the csv file containing tests (default: tests.csv)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help = "Provide the path to the output csv file (default: results.csv")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT_FILE, help = "Provide the path to a json file specifiying a system prompt to use")
    return parser.parse_args()
    
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

def write_results(file_path, results):
    file_exists = Path(file_path).exists()

    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "model", "test_description", "image_path", "prompt", "output", "expected_result", "check"])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)

def main():

    logger.info('Starting run.')

    args = parse_args()

    models_file = args.models
    output_file = args.output
    tests_file = args.tests
    system_prompts_file = args.system


    logger.info('Using model file: %s', models_file)
    logger.info('Using output_file file: %s', output_file)
    logger.info('Using tests_file file: %s', tests_file)
    
    models = load_models_list(models_file)
    tests = load_tests_json(tests_file)
    system_prompt = load_system_prompt(system_prompts_file)

    logger.info('Running %s tests against %s models.', len(tests), len(models))

    # Initialize where we're going to be storing the results of all tests
    results = []

    # iterate over the models we're going to test
    for model_idx, model_path in enumerate(models, start=1):

        logger.info('Starting run with model %s of %s: %s', model_idx, len(models), model_path)

        timestamp = datetime.now().isoformat(timespec="seconds")
        model, processor, config = load_model(model_path)

        # iterate over the tests we're going to test
        # each test is one test, prompt, image_directory, expected result
        # iterate over the tests by running the prompt against each image in the image directory
        for test_idx, test in enumerate(tests, start=1):

            logger.info('Starting test %s/%s: %s', test_idx, len(tests), test['test_description'])
            logger.info('Test %s/%s: prompt: %s', test_idx, len(tests), test['prompt'])
            logger.info('Test %s/%s: total eval strings: %s', test_idx, len(tests), len(test['expected_result']))

            # next, iterate over the prompts we're going to test

            # we will need to run each prompt over each image
            # next, iterate over the images we need to test against

            # get all the images
            test_images = load_test_images(test['image_directory'])

            # get all the check strings
            check_strings = test['expected_result']


            for image_idx, image in enumerate(test_images, start=1):

                image_path = join(test['image_directory'], image)

                logger.info('Test %s/%s: Running prompt on image %s/%s: %s', test_idx, len(tests), image_idx, len(test_images), image_path)

                # prompt = test['prompt']

                # Append the test prompt to the system prompt 
                # TODO: Can't figure out if the generator for these models will accept a system prompt as a kwarg

                prompt = system_prompt + "\n" + test['prompt']
                # print("Using prompt: %s", prompt)           

                logging.debug('Using prompt: %s', prompt)
                register_heif_opener()
                image = [Image.open(image_path)]
                this_result = []

                try:
                    # run the prompt against the image
                    output = run_prompt(model, processor, config, prompt, image)                                        
                    results.append(this_result)
                    
                    logger.info('Test %s/%s: image %s/%s: %s', test_idx, len(tests), image_idx, len(test_images), output.text)

                except Exception as e:
                    this_result = [{"model": model_path, "output": f"Error {e}"}]
                    results.append(this_result)

                # in this approach, we want one row per test, and to return True if any of the expected results are in the prompt output
                # so we just take output.text and the list of expected_results


                # This here is the test loop, it should probably be a function?
                test_satisfied = False

                for check in check_strings:
                    if str(check).lower() in output.text.lower():
                        test_satisfied = True
                        break
                
                this_result = [{"timestamp": timestamp, "model": model_path, "test_description": test['test_description'], "image_path": image_path, "prompt": test['prompt'], "output": output.text, "expected_result": test['expected_result'], "check": test_satisfied}]
                results.append(this_result)                
                write_results(output_file, this_result)

    logger.info('Run ended.')

    print(f"Results saved to: {output_file}")

    logging.shutdown()

logger = logging.getLogger(__name__)


stream_handler = logging.StreamHandler()

logging.basicConfig(
        format = ('%(asctime)s %(levelname)s: %(message)s'),
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers = [
            logging.FileHandler(DEFAULT_LOGFILE),
            stream_handler],
        encoding='utf-8', 
        level=logging.DEBUG)
    



if __name__ == "__main__":

    main()
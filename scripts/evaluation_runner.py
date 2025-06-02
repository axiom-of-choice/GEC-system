import logging
import sys
from gec.inference import Llama3InferenceEngine, T5InferenceEngine
from config.constants import *
from gec.evaluator import Evaluator
from datasets import load_from_disk
import os
import asyncio
import argparse

T5_FINETUNED_MODEL_DIR = os.path.join(os.getcwd(), "models/finished")


# Create logger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove all handlers associated with the root logger object (avoid duplicate logs)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create handler that outputs to notebook cell
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(description="GEC System CLI")
    parser.add_argument('--engine', choices=['llama', 't5'], required=True, help="Choose inference engine: llama or t5")
    parser.add_argument('--n_samples', type=int, default=None, help="Number of samples to evaluate (default: 50)")
    parser.add_argument('--predicted_dataset', type=str, default=None, help="Path to the predicted dataset (optional)")
    args = parser.parse_args()
    
    run_inference = True
    if args.predicted_dataset:
        if not os.path.exists(args.predicted_dataset):
            logger.error(f"Predicted dataset path {args.predicted_dataset} does not exist.")
            run_inference = True
        else:
            preprocessed_dataset = load_from_disk(args.predicted_dataset)
            if 'test' not in preprocessed_dataset:
                logger.error("Predicted dataset must contain a 'test' split.")
                run_inference = True
            else:
                logger.info(f"Using provided predicted dataset from {args.predicted_dataset}.")
                run_inference = False
    
    if run_inference:
        preprocessed_dataset = load_from_disk(os.path.join(FCE_DOWNLOAD_DATASET_DIR, "preprocessed_fce_dataset"))
    
    if args.engine == 'llama':
        llama3_engine = Llama3InferenceEngine(
            model_endpoint=LLAMA3_ENDPOINT,
            prompt_path=GENERAL_PROMPT_PATH
        )
        llama3_evaluator = Evaluator(preprocessed_dataset, llama3_engine, n_samples=args.n_samples)
        asyncio.run(llama3_evaluator.evaluate_accuracy_async())
        asyncio.run(llama3_evaluator.evaluate_gleu_async())
    elif args.engine == 't5':
        t5_engine = T5InferenceEngine(model_dir=T5_FINETUNED_MODEL_DIR, max_length=650)
        t5_evaluator = Evaluator(preprocessed_dataset, t5_engine, n_samples=args.n_samples)
        t5_evaluator.evaluate_accuracy()
        t5_evaluator.evaluate_gleu()

if __name__ == "__main__":
    main()

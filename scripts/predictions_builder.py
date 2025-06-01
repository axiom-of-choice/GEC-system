from gec.inference import T5InferenceEngine, Llama3InferenceEngine
from config.constants import MEDICAL_PROMPT_PATH, GENERAL_PROMPT_PATH, LLAMA3_ENDPOINT, FINAL_MODEL_DIR, FCE_DOWNLOAD_DATASET_DIR
import argparse
from datasets import load_from_disk, Dataset, DatasetDict
import os
import logging
import asyncio

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

def main():
    parser = argparse.ArgumentParser(description="CLI for Prediction")
    parser.add_argument("data", choices=["medical", "fce"], help="Data to predict on")
    parser.add_argument("engine", choices=["t5", "llama"], help="Inference engine type")
    parser.add_argument("save_to", help="Path to save the predictions")
    args = parser.parse_args()

    if args.engine == "t5":
        prediction_engine = T5InferenceEngine(model_dir=FINAL_MODEL_DIR, max_length=650)
    elif args.engine == "llama":
        prompt_path = MEDICAL_PROMPT_PATH if args.data == "medical" else GENERAL_PROMPT_PATH
        prediction_engine = Llama3InferenceEngine(model_endpoint=LLAMA3_ENDPOINT, prompt_path=prompt_path)
        # prediction_engine.logger.error("LLAMA3 endpoint does not support full batch inference yet. Please use T5 for batch predictions.")
    else:
        raise ValueError("Unsupported engine type. Choose 't5' or 'llama'.")
        
        
    if args.data == "fce":
        preprocessed_dataset = load_from_disk(os.path.join(FCE_DOWNLOAD_DATASET_DIR, "fce/preprocessed_fce_dataset"))
    elif args.data == "medical":
        import pandas as pd
        medical_df = pd.read_csv("/Users/isaac/Developer/GEC-system/data/data.csv")  # columns should be ['source', 'target']
        medical_df.rename(columns={'incorrect_sentence': 'source', 'correct_sentence': 'target'}, inplace=True)

        # Convert to HuggingFace Dataset
        medical_data = Dataset.from_pandas(medical_df)
        preprocessed_dataset = DatasetDict({'test': medical_data})
    else:
        raise ValueError("Unsupported data type. Choose 'medical' or 'fce'.")
    
    print("Starting prediction...")
    print(f"Dataset loaded with {len(preprocessed_dataset['test'])} samples.")
    if args.engine == "t5":
        prediction_engine.logger.info("Using T5InferenceEngine for predictions.")
        prediction = prediction_engine.batch_correct(preprocessed_dataset['test']['source'])
    else:
        prediction_engine.logger.info("Using Llama3InferenceEngine for predictions.")
        prediction = asyncio.run(prediction_engine.async_batch_correct(preprocessed_dataset['test']['source'], max_concurrent=5))
    print("Prediction completed successfully.")
    # prediction_df = preprocessed_dataset['test'].to_pandas().drop(columns=['source', 'target'], errors='ignore')
    # prediction_df.to_csv(args.save_to, index=False)
    prediction_df = Dataset.from_dict({
        'source': preprocessed_dataset['test']['source'],
        'target': preprocessed_dataset['test']['target'],
        'prediction': prediction
    }).to_pandas()
    prediction_df.to_csv(args.save_to, index=False)

if __name__ == "__main__":
    main()
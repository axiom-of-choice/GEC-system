from transformers import PreTrainedTokenizer
from datasets import DatasetDict
from typing import Dict
import logging
from typing import List

class T5Preprocessor:
    """
    T5 Preprocessor for the data. Could be implemented as abstract class to have multiple preprocessors.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_lenght (int): The maximum lenght of the input. Default is 512.
        truncation (bool): Whether to truncate the input. Default is True.
        padding (str): The padding to use. Default is "max_length".

    """


    def __init__(self, tokenizer: PreTrainedTokenizer, truncation: bool = True, padding: str = "max_length"):
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.padding = padding
        self.logger = logging.getLogger(self.__class__.__name__)

    def _preprocess_function(self, examples: Dict[str,str], max_length: int) -> Dict[str, List[List[int]]]:
        """
        Preprocess examples tokenizing them using the instance parameters.
        """
        inputs = ["correct grammar: " + s for s in examples['source']]
        targets = [t for t in examples['target']]
        model_inputs = self.tokenizer(inputs, max_length=max_length, truncation=self.truncation, padding=self.padding)
        labels = self.tokenizer(targets, max_length=max_length, truncation=self.truncation, padding=self.padding)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    @staticmethod
    def _get_max_input_length(dataset: DatasetDict) -> int:
        """
        Get the maximum input length in the dataset.
        """
        max_length = 0
        for split in dataset:
            split_max = max(len(x) for x in dataset[split]['source'])
            #self.logger.info(f"Max input_ids length in {split}: {split_max}")
            max_length = max(max_length, split_max)
            #self.logger.info(f"Overall max input_ids length: {max_length}")
        return max_length
    
    
    def save_tokenized_dataset(self, dataset: DatasetDict, output_dir: str) -> None:
        """
        Save the tokenized dataset to disk.
        """
        self.logger.info(f"Saving tokenized dataset to {output_dir}...")
        dataset.save_to_disk(output_dir)
        self.logger.info("Tokenized dataset saved successfully.")

    def preprocess(self, dataset: DatasetDict, max_length: int) -> DatasetDict:
        """
        Preprocess the dataset.
        """
        self.logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(lambda examples: self._preprocess_function(examples, max_length), batched=True)
        self.logger.info("Dataset preprocessed")
        return tokenized_dataset
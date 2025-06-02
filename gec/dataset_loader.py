### Dataset Loader class handling the dataset creation
import tarfile
from datasets import Dataset, DatasetDict
import requests
from typing import Optional, List, Dict, Tuple
from config.constants import *
import logging


class M2DatasetLoader:
    """
    Loader for datasets in M2 format. Right now it's only tested for BCE2019

    Args:
        dataset_dir (str): Directory where the dataset is stored. Default is FCE_DATASET_DIR.
        train_file (str): Name of the training file. Default is FCE_TRAIN_NAME.
        dev_file (str): Name of the development file. Default is FCE_DEV_NAME.
        test_file (str): Name of the test file. Default is FCE_TEST_NAME.
        dataset_url (str): URL to download the dataset from. Default is FCE_URL.
        fce_download_dir (str): Directory to download the dataset to. Default is FCE_DOWNLOAD_DATASET_DIR.
    """
    def __init__(self,dataset_dir: str = FCE_DATASET_DIR, train_file: str = FCE_TRAIN_NAME,
                 dev_file: str = FCE_DEV_NAME, test_file: str = FCE_TEST_NAME, dataset_url: str = FCE_URL,
                 fce_download_dir: str = FCE_DOWNLOAD_DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.fce_download_dir = fce_download_dir
        self.train_file = os.path.join(dataset_dir, train_file)
        self.dev_file = os.path.join(dataset_dir, dev_file)
        self.test_file = os.path.join(dataset_dir, test_file)
        self.url = dataset_url
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = None

    def download_and_extract(self) -> None:
        """
        Download the file and extract it into a directory if it does not exist
        """
        if not os.path.exists(self.fce_download_dir):
            os.makedirs(self.fce_download_dir, exist_ok=True)
            self.logger.info(f"Downloading BEA 2019 dataset...")
            tar_path = os.path.join(self.fce_download_dir, "bea19.tar.gz")
            with requests.get(self.url, stream=True) as r:
                with open(tar_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.logger.info("Extracting dataset...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=self.fce_download_dir)
            os.remove(tar_path)
            self.logger.info("BEA dataset downloaded and extracted successfully.")
        else:
            self.logger.warning("BEA dataset already exists locally.")

    def load_dataset(self) -> DatasetDict:
        """ Create the dataset in the Transformers format """
        dataset = DatasetDict({
            'train': Dataset.from_list(self._parse_m2_file(self.train_file)),
            'validation': Dataset.from_list(self._parse_m2_file(self.dev_file)),
            'test': Dataset.from_list(self._parse_m2_file(self.test_file))
        })
        self.logger.info(f"Loaded BEA dataset: {len(dataset['train'])} train, {len(dataset['validation'])} dev, {len(dataset['test'])} test")
        self.dataset = dataset
        return dataset
    
    def save_dataset(self, output_dir: str = os.path.join(os.getcwd(), FCE_DOWNLOAD_DATASET_DIR)) -> None:
        """
        Parses the M2 files and saves the resulting DatasetDict to disk in HuggingFace format.
        Args:
            output_dir (str): Directory to save the dataset.
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        self.dataset.save_to_disk(output_dir + "parsed_fce_dataset")
        self.logger.info(f"Saved parsed dataset to {output_dir}")

    def _parse_m2_file(self, filepath: str) -> List[Dict[str, str]]:
        """
        Parses an M2 file and returns a list of {source, target} dictionaries,
        where each 'target' corresponds to one annotator's corrections.
        """
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sentence = ""
        edits_by_annotator = dict()

        for line in lines + ['\n']:  # Add sentinel newline
            line = line.strip()
            if line.startswith(SENTENCE_TAG):
                if sentence:
                    if edits_by_annotator:
                        for annotator_id, edits in edits_by_annotator.items():
                            corrected = self._apply_m2_edits(sentence, edits)
                            data.append({'source': sentence, 'target': corrected})
                    else:
                        data.append({'source': sentence, 'target': sentence})
                sentence = line[2:]
                edits_by_annotator = dict()
            elif line.startswith(ANNOTATION_TAG):
                parts = line[2:].split(SEP)
                span = list(map(int, parts[0].split()))
                error_type = parts[1]
                correction = parts[2]
                annotator_id = int(parts[-1])
                if annotator_id not in edits_by_annotator:
                    edits_by_annotator[annotator_id] = []
                edits_by_annotator[annotator_id].append((span, correction, error_type))
            elif line == "" and sentence:
                if edits_by_annotator:
                    for annotator_id, edits in edits_by_annotator.items():
                        corrected = self._apply_m2_edits(sentence, edits)
                        data.append({'source': sentence, 'target': corrected})
                else:
                    data.append({'source': sentence, 'target': sentence})
                sentence = ""
                edits_by_annotator = dict()
        return data

    def _apply_m2_edits(self, sentence: str, edits: list):
        """
        Applies M2 format edits to the original sentence.
        :param sentence: Original sentence (string)
        :param edits: List of (span, correction, error_type)
        :return: Corrected sentence
        """
        tokens = sentence.strip().split()
        offset = 0
        for (span, correction, error_type) in edits:
            start, end = span
            if error_type == NO_EDIT_TAG or (start == -1 and end == -1):
                continue
            # Adjust indices by current offset
            start_adj = start + offset
            end_adj = end + offset
            correction_tokens = correction.strip().split() if correction.strip() else []
            tokens = tokens[:start_adj] + correction_tokens + tokens[end_adj:]
            offset += len(correction_tokens) - (end - start)
        return ' '.join(tokens)
    
    @staticmethod
    def most_common_edit_types(gold_m2_path: str, n: int = 10) -> List[Tuple[str, int]]:
        """
        Returns the most common edit types in the gold M2 file.
        :param gold_m2_path: Path to the gold M2 file.
        :param n: Number of most common edit types to return.
        :return: List of tuples (edit_type, count).
        """
        from collections import Counter
        edit_types = Counter()
        
        with open(gold_m2_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(ANNOTATION_TAG):
                    parts = line[2:].split(SEP)
                    error_type = parts[1]
                    edit_types[error_type] += 1
        
        return edit_types.most_common(n)

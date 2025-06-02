from sklearn.metrics import accuracy_score
from nltk.translate.gleu_score import sentence_gleu
from typing import Optional
import re
from inference import BaseInferenceEngine
import logging
from typing import List, Tuple
from datasets import DatasetDict

class Evaluator:
    """ Evaluator class for evaluating the performance of the inference engine on a dataset.
    Args:
        dataset (DatasetDict): The dataset to evaluate.
        inference_engine (BaseInferenceEngine): The inference engine to use for evaluation.
        n_samples (Optional[int]): Number of samples to evaluate. If None, evaluates the entire test set. Default is None.
    """
    def __init__(self, dataset, inference_engine: BaseInferenceEngine, n_samples: Optional[int] = None, predicted_dataset: Optional[DatasetDict] = None):
        self.dataset = dataset
        self.engine = inference_engine
        self.n_samples = n_samples
        self.test_data, self.references, self.predictions = None, None, None
        if predicted_dataset is not None and isinstance(predicted_dataset, DatasetDict):
            self.test_data = predicted_dataset['test']['source']
            self.references = predicted_dataset['test']['target']
            self.predictions = predicted_dataset['test']['prediction']
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def normalize_text(text):
        # Remove spaces before punctuation and ensure single space after
        text = re.sub(r'\s+([.,!?;:"])', r'\1', text)
        text = re.sub(r'([.,!?;:"])([^\s])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_samples(self) -> Tuple[List[str], List[str], List[str]]:
        """ Fetch samples from the test set."""
        test_data = self.dataset['test']
        n = min(self.n_samples, len(test_data)) if (self.n_samples is not None and self.n_samples > 0) else len(test_data)
        self.logger.info(f"Fetching {n} samples from the test set.")
        sample_sentences = [test_data[i]['source'] for i in range(n)]
        references = [test_data[i]['target'] for i in range(n)]
        predictions = self.engine.batch_correct(sample_sentences)
        return sample_sentences, references, predictions
    
    def  _get_samples_if_not_available(self):
        """
        Check if samples are available.
        """
        if self.test_data is None or self.references is None or self.predictions is None:
            self.logger.warning("Samples not available. Fetching samples...")
            self.test_data, self.references, self.predictions = self._get_samples()
        else:
            self.logger.info("Samples already available. Using cached samples.")

    def evaluate_accuracy(self) -> float:
        """ 
        Evaluate the accuracy of the predictions against the references.
        It computes the exact match accuracy.
        Returns:
            float: The exact match accuracy.
        """
        self._get_samples_if_not_available()
        norm_refs = [self.normalize_text(str(ref)) for ref in self.references]
        norm_preds = [self.normalize_text(str(pred)) for pred in self.predictions]
        accuracy = accuracy_score(norm_refs, norm_preds)
        self.logger.info(f"Exact match accuracy on test set: {accuracy:.4f}")
        return accuracy

    def evaluate_gleu(self):
        """
        Evaluate the GLEU score of the predictions against the references.
        It uses the nltk library to compute the GLEU score.
        Returns:
            float: The average GLEU score across all samples.
            
        """
        self._get_samples_if_not_available()
        gleu_scores = []
        for pred, ref in zip(self.predictions, self.references):
            ref_tokens = self.normalize_text(str(ref)).split()
            pred_tokens = self.normalize_text(str(pred)).split()
            gleu = sentence_gleu([ref_tokens], pred_tokens)
            gleu_scores.append(gleu)
        avg_gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0
        self.logger.info(f"Average GLEU score on test set: {avg_gleu:.4f}")
        return avg_gleu

    # --- ASYNC BATCH INFERENCE FOR LLAMA3 ---
    async def evaluate_accuracy_async(self)-> Optional[float]:
        """ Evaluate the accuracy of the predictions against the references using async batch inference.
        It computes the exact match accuracy.
        This method is specifically designed for engines that support async batch inference.

        Returns:
            Optional[float]: The exact match accuracy if async batch inference is available, otherwise None.
        """
        # Fetch samples (sentences and references)
        test_data = self.dataset['test']
        n = min(self.n_samples, len(test_data)) if (self.n_samples is not None and self.n_samples > 0) else len(test_data)
        self.logger.info(f"Fetching {n} samples from the test set (async).")
        sample_sentences = [test_data[i]['source'] for i in range(n)]
        references = [test_data[i]['target'] for i in range(n)]
        # Use async batch correct if available
        if hasattr(self.engine, "async_batch_correct"):
            predictions = await self.engine.async_batch_correct(sample_sentences)
            self.test_data, self.references, self.predictions = sample_sentences, references, predictions
            norm_refs = [self.normalize_text(str(ref)) for ref in self.references]
            norm_preds = [self.normalize_text(str(pred)) for pred in self.predictions]
            accuracy = accuracy_score(norm_refs, norm_preds)
            self.logger.info(f"Exact match accuracy on test set: {accuracy:.4f}")
            return accuracy
        else:
            self.logger.warning("Async batch inference not available for this engine.")
            return None

    async def evaluate_gleu_async(self)-> Optional[float]:
        """ Evaluate the GLEU score of the predictions against the references using async batch inference.
        It uses the nltk library to compute the GLEU score.
        This method is specifically designed for engines that support async batch inference.

        Returns:
            Optional[float]: The average GLEU score across all samples if async batch inference is available, otherwise None.
        """
        # Fetch samples (sentences and references)
        test_data = self.dataset['test']
        n = min(self.n_samples, len(test_data)) if (self.n_samples is not None and self.n_samples > 0) else len(test_data)
        self.logger.info(f"Fetching {n} samples from the test set (async).")
        sample_sentences = [test_data[i]['source'] for i in range(n)]
        references = [test_data[i]['target'] for i in range(n)]
        # Use async batch correct if available
        if hasattr(self.engine, "async_batch_correct"):
            predictions = await self.engine.async_batch_correct(sample_sentences)
            self.test_data, self.references, self.predictions = sample_sentences, references, predictions
            gleu_scores = []
            for pred, ref in zip(self.predictions, self.references):
                ref_tokens = self.normalize_text(str(ref)).split()
                pred_tokens = self.normalize_text(str(pred)).split()
                gleu = sentence_gleu([ref_tokens], pred_tokens)
                gleu_scores.append(gleu)
            avg_gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0
            self.logger.info(f"Average GLEU score on test set: {avg_gleu:.4f}")
            return avg_gleu
        else:
            self.logger.warning("Async batch inference not available for this engine.")
            return None
        
    def evaluate_single(self, pred, ref):
        norm_pred = Evaluator.normalize_text(pred)
        norm_ref = Evaluator.normalize_text(ref)
        exact = int(norm_pred == norm_ref)
        gleu = sentence_gleu([norm_ref.split()], norm_pred.split())
        return {"exact_match": exact, "gleu": gleu}
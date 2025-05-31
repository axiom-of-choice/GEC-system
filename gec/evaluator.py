from sklearn.metrics import accuracy_score
from nltk.translate.gleu_score import sentence_gleu
from typing import Optional
import re
from inference import BaseInferenceEngine
import logging
from typing import List, Tuple


class Evaluator:
    def __init__(self, dataset, inference_engine: BaseInferenceEngine, n_samples: Optional[int] = None):
        self.dataset = dataset
        self.engine = inference_engine
        self.n_samples = n_samples
        self.test_data, self.references, self.predictions = None, None, None
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
        predictions = [self.engine.correct_sentence(sentence) for sentence in sample_sentences]
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

    def evaluate_accuracy(self):
        self._get_samples_if_not_available()
        norm_refs = [self.normalize_text(ref) for ref in self.references]
        norm_preds = [self.normalize_text(pred) for pred in self.predictions]
        accuracy = accuracy_score(norm_refs, norm_preds)
        print(f"Exact match accuracy on test set: {accuracy:.4f}")
        return accuracy

    def evaluate_gleu(self):
        self._get_samples_if_not_available()
        gleu_scores = []
        for pred, ref in zip(self.predictions, self.references):
            ref_tokens = self.normalize_text(ref).split()
            pred_tokens = self.normalize_text(pred).split()
            gleu = sentence_gleu([ref_tokens], pred_tokens)
            gleu_scores.append(gleu)
        avg_gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0
        print(f"Average GLEU score on test set: {avg_gleu:.4f}")
        return avg_gleu
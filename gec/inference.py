from abc import ABC, abstractmethod
import logging
import sys
import os
import requests
from typing import Dict, Union
import asyncio
import aiohttp

# constants, ideally defined in other file and imported if done in a github repo

FCE_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"
FCE_DEV_NAME = "fce.dev.gold.bea19.m2"
FCE_TRAIN_NAME = "fce.train.gold.bea19.m2"
FCE_TEST_NAME = "fce.test.gold.bea19.m2"
FCE_DOWNLOAD_DATASET_DIR = os.path.join(os.getcwd(), "data")
FCE_DATASET_DIR = os.path.join(os.getcwd(), "data/fce/m2")
SENTENCE_TAG = "S "
ANNOTATION_TAG = "A "
NO_EDIT_TAG = "noop"
SEP = "|||"
## We're gonna use T5 small for this task because we're just correcting spelling and we don't wait to wait hours for training.
MODEL_NAME = "t5-small"
FINETUNED_MODEL_OUTPUT_DIR = "./t5_finetuned"
PROMPT_PATH = os.path.join(os.getcwd(), "config/prompt.txt")
LLAMA3_ENDPOINT = "http://127.0.0.1:11434/api/generate"
TEXT_TO_REPLACE_IN_PROMPT = "<text_to_replace>"


class BaseInferenceEngine(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    @abstractmethod
    def correct_sentence(self, sentence: str) -> str:
        pass
    
    
from typing import Union
import json as _json

class Llama3InferenceEngine(BaseInferenceEngine):
    model_name: str = "llama3"
    stream: bool = False
    response_format: dict = {
        "type": "object",
        "properties": {
            "original_text": {"type": "string"},
            "corrected_text": {"type": "string"}
        },
        "required": ["original_text", "corrected_text"]
    }
    
    def __init__(self, model_endpoint: str, prompt_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_endpoint = model_endpoint
        self.prompt_path = prompt_path
        self.prompt = self._parse_prompt()
        
    def _parse_prompt(self) -> str:
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        if not prompt:
            self.logger.error("Prompt is empty. Please check the prompt file.")
            raise ValueError("Prompt is empty. Please check the prompt file.")
        return prompt
    
    def _replace_prompt_variables(self, sentence: str) -> str:
        return self.prompt.replace(f"{TEXT_TO_REPLACE_IN_PROMPT}", sentence)

    def send_correct_request(self, sentence: str) -> Dict[str, Union[str, Dict[str, str]]]:
        prompt = self._replace_prompt_variables(sentence)
        try:
            response = requests.post(
                self.model_endpoint, 
                json={
                    "model": self.model_name,
                    "prompt": str(prompt),
                    "stream": self.stream,
                    "format": self.response_format
                }
            )
            response_data = response.json()
            if not response_data:
                self.logger.error("No corrected sentence returned from the model.")
                raise ValueError("No corrected sentence returned from the model.")
            if "response" in response_data and isinstance(response_data["response"], str):
                response_data["response"] = _json.loads(response_data["response"])
            return response_data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during model inference: {e}")
            raise RuntimeError(f"Error during model inference: {e}")
        
    def correct_sentence(self, sentence: str) -> str:
        self.logger.info(f"Correcting sentence: {sentence}")
        response = self.send_correct_request(sentence)
        if isinstance(response, dict):
            corrected_sentence = response.get("response", {}).get("corrected_text", "")
            self.logger.info(f"Corrected sentence: {corrected_sentence}")
            return corrected_sentence
        else:
            return ""

    # --- ASYNC BATCH INFERENCE ---
    async def async_correct_sentence(self, session, sentence: str) -> str:
        prompt = self._replace_prompt_variables(sentence)
        payload = {
            "model": self.model_name,
            "prompt": str(prompt),
            "stream": self.stream,
            "format": self.response_format
        }
        async with session.post(self.model_endpoint, json=payload) as resp:
            response_data = await resp.json()
            if not response_data:
                return ""
            if "response" in response_data and isinstance(response_data["response"], str):
                import json as _json
                response_data["response"] = _json.loads(response_data["response"])
            return response_data.get("response", {}).get("corrected_text", "")

    async def async_batch_correct(self, sentences):
        async with aiohttp.ClientSession() as session:
            tasks = [self.async_correct_sentence(session, s) for s in sentences]
            return await asyncio.gather(*tasks)


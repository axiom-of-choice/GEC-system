from abc import ABC, abstractmethod
import logging
import sys
import os
import requests
from typing import Dict, Union, Optional
import asyncio
import aiohttp
from transformers import T5ForConditionalGeneration, T5Tokenizer
from config.constants import *

class BaseInferenceEngine(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    @abstractmethod
    def correct_sentence(self, sentence: str) -> str:
        pass
    @abstractmethod
    def batch_correct(self, sentences: list, batch_size: int = 16) -> list:
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
        
    def batch_correct(self, sentences: list, batch_size: int = 16) -> list:
        """Not implemented for Llama3InferenceEngine as it does not support full batch inference yet.
        This method is a placeholder to maintain interface consistency with BaseInferenceEngine.

        Args:
            sentences (list): _description_
            batch_size (int, optional): _description_. Defaults to 16.

        Returns:
            list: _description_
        """
        return super().batch_correct(sentences, batch_size)

    # --- ASYNC BATCH INFERENCE ---
    async def async_correct_sentence(self, session, sentence: str) -> str:
        prompt = self._replace_prompt_variables(sentence)
        payload = {
            "model": self.model_name,
            "prompt": str(prompt),
            "stream": self.stream,
            "format": self.response_format
        }
        self.logger.info(f"Sending async request for sentence: {sentence}")
        async with session.post(self.model_endpoint, json=payload) as resp:
            response_data = await resp.json()
            if not response_data:
                return ""
            if "response" in response_data and isinstance(response_data["response"], str):
                import json as _json
                response_data["response"] = _json.loads(response_data["response"])
                corrected_text = response_data.get("response", {}).get("corrected_text", "")
                self.logger.info(f"Async corrected sentence: {corrected_text}")
            
                return corrected_text if corrected_text else ""
            else:
                self.logger.error("Response format is not as expected.")
                return ""

    async def async_batch_correct(self, sentences, max_concurrent=5):
        timeout = aiohttp.ClientTimeout(total=60)
        semaphore = asyncio.Semaphore(max_concurrent)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def sem_task(sentence):
                async with semaphore:
                    return await self.async_correct_sentence(session, sentence)
            tasks = [sem_task(s) for s in sentences]
            return await asyncio.gather(*tasks)



class T5InferenceEngine(BaseInferenceEngine):
    def __init__(self, model_dir: str, max_length: Optional[int] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length
        self.logger.info(f"T5 model loaded from {model_dir} with max length {self.max_length}")
        

    def correct_sentence(self, sentence:str) -> str:
        """
        Corrects a sentence using the T5 model.
        Args:
            sentence (str): The sentence to correct.
        Returns:
            str: The corrected sentence.
        """
        self.logger.info(f"Correcting sentence: {sentence}")
        inputs = self.tokenizer("correct grammar: " + sentence, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        outputs = self.t5_model.generate(**inputs, max_length=self.max_length)
        corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Corrected sentence: {corrected_sentence}")
        return corrected_sentence
    
    def batch_correct(self, sentences: list, batch_size: int = 16) -> list:
        """
        Corrects a batch of sentences using the T5 model.
        Args:
            sentences (list): List of sentences to correct.
            batch_size (int): Number of sentences per batch.
        Returns:
            list: List of corrected sentences.
        """
        corrected = []
        for i in range(0, len(sentences), batch_size):
            self.logger.info(f"Processing batch {i // batch_size + 1} with size {min(batch_size, len(sentences) - i)}")
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(
                ["correct grammar: " + s for s in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            outputs = self.t5_model.generate(**inputs, max_length=self.max_length)
            batch_corrected = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            corrected.extend(batch_corrected)
        self.logger.info(f"Batch correction completed. Total corrected sentences: {len(corrected)}")
        return corrected
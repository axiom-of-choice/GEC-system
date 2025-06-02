from typing import Tuple, Optional
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback, T5Tokenizer
from datasets import DatasetDict
import torch
import logging
from config.constants import *
import time


class T5Trainer:
    def __init__(self, model_name: str = MODEL_NAME, output_dir: str = FINETUNED_MODEL_OUTPUT_DIR,
                 logging_dir: str = "./finetune_logs", save_strategy: str = "epoch", resume_from_dir: Optional[str] = None,
                 batch_size: int = 8, mixed_precision: bool = False, early_stopping: bool = True,
                 eval_strategy: str = "epoch", early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0, 
                 metric_for_best_model: str = "eval_loss", greater_is_better: bool = False):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.save_strategy = save_strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.resume_from_dir = resume_from_dir
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.early_stopping = early_stopping
        self.eval_strategy = eval_strategy
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        

    def _check_if_gpu_available(self) -> bool:
        """
        Check if GPU is available.
        """
        available = torch.cuda.is_available()

        if available:
            self.logger.info("GPU available")
        else:
            self.logger.warning("GPU not available")

        return available

    def _check_if_model_already_available(self) -> bool:
        """
        Check if model is already available.
        """
        available = os.path.exists(self.output_dir)

        if available:
            self.logger.info("Model already available")
        else:
            self.logger.warning("Model not available")

        return available

    def _create_unique_dir_for_model(self) -> str:

        """
        Create a unique directory for the model based on the current time.
        """
        current_time = time.strftime("%Y%m%d-%H%M%S")
        unique_dir = os.path.join(self.output_dir, current_time)
        os.makedirs(unique_dir, exist_ok=True)
        self.logger.info(f"Created unique directory for model: {unique_dir}")
        return unique_dir


    def train(self, tokenized_dataset:DatasetDict, epochs=3, learning_rate: float = 3e-4, eval_strategy = "epoch") -> Tuple[str, T5ForConditionalGeneration, T5Tokenizer]:
        """
        Train the T5 model.
        Args:
            tokenized_dataset (DatasetDict): The tokenized dataset.
            epochs (int): The number of epochs to train for.
        Returns:
            Tuple[Model_Dir (ID), T5ForConditionalGeneration, T5Tokenizer]: The trained model and tokenizer.
        """
        self.logger.info("Checking if GPU is available...")
        self._check_if_gpu_available()
        #self.logger.info("Checking if model is already available...")
        #self._check_if_model_already_available()
        # Determine the output directory based on whether we're resuming or starting fresh
        if self.resume_from_dir and os.path.exists(self.resume_from_dir):
            self.output_dir = self.resume_from_dir
            self.logger.info(f"Resuming training from directory: {self.output_dir}")
        else:
             self.output_dir = self._create_unique_dir_for_model()
             self.logger.info(f"Starting new training in directory: {self.output_dir}")

        # self.output_dir = self._create_unique_dir_for_model()


        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=epochs,
            eval_strategy = self.eval_strategy,
            save_strategy=self.save_strategy,
            logging_dir=self.logging_dir,
            learning_rate=learning_rate,
            report_to="none", #Needed to avoid wandb api key request.,
            # fp16=self.mixed_precision,  # Enable mixed precision training if specified
        )
        
        if self.early_stopping:
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = self.metric_for_best_model
            training_args.greater_is_better = self.greater_is_better
            training_args.save_total_limit = 1
            training_args.eval_strategy = self.eval_strategy
        self.logger.info("Training arguments set")
        
        if self.mixed_precision:
            training_args.fp16 = True
            self.logger.info("Mixed precision training enabled")


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation']
        )
        
        if self.early_stopping:
            trainer.add_callback(EarlyStoppingCallback(self.early_stopping_patience, self.early_stopping_threshold))
            self.logger.info("Early stopping callback added")

          # Resume training if a checkpoint exists in the output directory
        if self.resume_from_dir and os.path.exists(self.resume_from_dir):
             # The Trainer class automatically handles resuming from a directory if it exists
             # and contains a checkpoint. You don't need to explicitly load the state dicts
             # if you are using the Trainer's resume functionality.
             trainer.train(resume_from_checkpoint=self.output_dir)
        else:
            trainer.train()


        self.logger.info("T5 model trained")
        trainer.save_model(self.output_dir)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return self.output_dir, self.model, self.tokenizer
# constants, ideally defined in other file and imported if done in a github repo
import os

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
MEDICAL_PROMPT_PATH = os.path.join(os.getcwd(), "config/prompt_medical.txt")
GENERAL_PROMPT_PATH = os.path.join(os.getcwd(), "config/prompt_general.txt")
LLAMA3_ENDPOINT = "http://127.0.0.1:11434/api/generate"
TEXT_TO_REPLACE_IN_PROMPT = "<text_to_replace>"
FINAL_MODEL_DIR = os.path.join(os.getcwd(), "models/improved")

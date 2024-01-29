from transformers import GPT2Config, GPT2Tokenizer
from GPT2SP.GPT2ForSequenceClassification import GPT2ForSequenceClassification as GPT2SP
import torch
import os
from utils import download_file_from_gcs
from constants import MODEL_ARTIFACTS_BUCKET, WEIGHTS_FILE_NAME, MODEL_STORE_PATH

class EstimateModelHandler:
    _instance = None
    _models = {}
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EstimateModelHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) :
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._tokenizer.pad_token = "[PAD]"

    def get_model(self, organization_id):
        """Retrieve a model for a given organization ID, if it exists."""
        model = self._models.get(organization_id, None)
        if model is None:
            model = self.__initialize_new_model(organization_id)
        return model
    
    def get_tokenizer(self):
        return self._tokenizer
    
    def clear_model(self, organization_id):
        """Clear a model for a given organization ID, if it exists."""
        if organization_id in self._models:
            del self._models[organization_id]
    
    def clear_all_models(self):
        """Clear all models."""
        self._models = {}


    def __set_model(self, organization_id, model):
        """Set a model for a given organization ID."""
        self._models[organization_id] = model
    
    def __get_weights_path(self,organization_id):
        if os.path.exists(f'{MODEL_STORE_PATH}/{organization_id}/{WEIGHTS_FILE_NAME}'):
            return f'{MODEL_STORE_PATH}/{organization_id}/{WEIGHTS_FILE_NAME}'
        else:
            return None
    
    def __initialize_new_model(self, organization_id):
        config = GPT2Config(num_labels=1, pad_token_id=50256)
        gpt2sp = GPT2SP.from_pretrained('gpt2', config=config)

        state_dict_path = self.__get_weights_path(organization_id)

        if state_dict_path is None:
            # no trained model found in local check in the storage
            result =  download_file_from_gcs(MODEL_ARTIFACTS_BUCKET, f'{organization_id}/{WEIGHTS_FILE_NAME}', f'{MODEL_STORE_PATH}/{organization_id}/{WEIGHTS_FILE_NAME}')
            if result:
                state_dict_path = f'{MODEL_STORE_PATH}/{organization_id}/{WEIGHTS_FILE_NAME}'
            else:
                return None
            #Can't find the training artifacts

        state_dict = torch.load(state_dict_path)
        gpt2sp.load_state_dict(state_dict)
        gpt2sp.to('cpu')
        gpt2sp.eval()

        self.__set_model(organization_id, gpt2sp)
        return self.get_model(organization_id)
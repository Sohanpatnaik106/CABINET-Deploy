
import torch
import torch.nn as nn
from transformers import BartConfig, T5ForConditionalGeneration, T5Model, T5Config
from .modeling_bart import HighlightedCluBartForConditionalGeneration

class HighlightedCluBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(HighlightedCluBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = HighlightedCluBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, 
                decoder_input_ids: torch.LongTensor = None, highlighted_cells: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, 
                                 decoder_input_ids = decoder_input_ids, highlighted_cells = highlighted_cells)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs


class T5ModelForTableReasoning(nn.Module):
    
    def __init__(self, config):
        super(T5ModelForTableReasoning, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained("google/flan-t5-xl")

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits


class T5ModelForTableCellHighlighting(nn.Module):
    
    def __init__(self, config):
        super(T5ModelForTableCellHighlighting, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained("google/flan-t5-xl")

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits
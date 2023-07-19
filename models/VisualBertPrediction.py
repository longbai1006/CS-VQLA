import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig
import torch.nn.functional as F
from utils.vqla import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualBertPrediction(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self):
        super(VisualBertPrediction, self).__init__()
        VBconfig = VisualBertConfig(vocab_size = 30522, visual_embedding_dim = 512, num_hidden_layers = 6, num_attention_heads = 8, hidden_size = 2048)
        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.bbox_embed = MLP(2048, 2048, 4, 3)

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertEncoder(**inputs)
                
        return {
            'features': outputs['pooler_output'],
        }
def visualbert():
    model = VisualBertPrediction()
    return model
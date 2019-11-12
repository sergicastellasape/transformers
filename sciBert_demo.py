import torch
from transformers import *


input_sentence = "this is a shorter test"

# model and tokenizer class for the concrete downstream task: BertModel, BertForQuestionAnswering, etc.
model_class = BertModel
tokenizer_class = BertTokenizer

# weights to use
pretrained_weights = 'scibert-scivocab-uncased'

# initialize models and tokenizer
model = model_class.from_pretrained(pretrained_weights)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# generate input in the proper data format
input_tokens = tokenizer.encode(input_sentence)
input_tensor = torch.tensor([input_tokens])

# run the model forward
model_output = model(input_tensor)

# store states & attentions
final_attention, final_state, hidden_states_tuple, hidden_attentions_tuple = model_output

print(model_output[0].shape)
print(model_output[1].shape)
print(len(model_output[2]))
print(len(model_output[3]))

# Final attention: tensor [batch, seq_length, embedding_size]
# Final state: tensor.Size[batch, embedding_size]
# hidden_states_tuple: tuple of length # layers + 1 (it probably inclues the input)
    # hidden_states_tuple[i]: tensor.Size(batch, seq_length, embedding_size)
# hidden_attentions_tuple: tuple of length # layers
    # hidden_attentions_tuple[i]: tensor.Size(batch, #heads, seq_length, seq_length)
    
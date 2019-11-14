import stanfordnlp
import torch
from transformers import *

stanfordnlp.download('en') # download english models that the neural pipeline will uses
nlp = stanfordnlp.Pipeline() # setting a default neural pipeline for english
# Now nlp is a function that receives a string as input and returns an nlp object

# create tokenizer
pretrained_weights = 'scibert-scivocab-uncased'
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

inp_string = "The data for our first experiment is a corpus of parsed sentences from the Penn Treebank"
inp_tokens = tokenizer.encode(inp_string)

sequence = nlp(inp_string)
sequence.sentences[0].print_dependencies()
#print(f'Sentence length is: {len(sequence)}')
#print(f'Tokenized length is: {len(inp_tokens)}')
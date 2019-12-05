import torch
from transformers import *
from custom.transformer_sentence import TransformerSentence
from tqdm import tqdm
import re


def load_dataset(txt_path=None,
                return_embeddings=False,
                MODEL=BertModel.from_pretrained('scibert-scivocab-uncased'),
                TOKENIZER=BertTokenizer.from_pretrained('scibert-scivocab-uncased')):
                
    if txt_path is None:
        raise ValueError("txt_path must be specified as a named argument! \
            E.g. txt_path=../dataset/yourfile.txt")

    # Read input sequences from .txt file and put them in a list
    with open(txt_path) as f:
        text = f.read()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    try:
        sentences.remove('') # remove possible empty strings
    except:
        None
    
    list_SentenceObj, ALL_INITIAL_EMBEDDINGS, ALL_CONTEXT_EMBEDDINGS = [], [], []
    
    for raw_sentence in tqdm(sentences):
        SentenceObj = TransformerSentence(raw_sentence,
                                          model=MODEL,
                                          tokenizer=TOKENIZER)
        SentenceObj.write_summary(print_tokens=False)
        list_SentenceObj.append(SentenceObj)
        ALL_INITIAL_EMBEDDINGS.append(SentenceObj.summary['states'][0, :, :])
        ALL_CONTEXT_EMBEDDINGS.append(SentenceObj.summary['states'][-1, :, :])

    ALL_INITIAL_EMBEDDINGS = torch.cat(ALL_INITIAL_EMBEDDINGS, dim=0)
    ALL_CONTEXT_EMBEDDINGS = torch.cat(ALL_CONTEXT_EMBEDDINGS, dim=0)
    
    if return_embeddings:
        return sentences, list_SentenceObj, ALL_INITIAL_EMBEDDINGS, ALL_CONTEXT_EMBEDDINGS
    else:
        return sentences, list_SentenceObj
    
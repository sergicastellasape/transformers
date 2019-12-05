from transformers import *
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
import math
import matplotlib.pyplot as plt


class TransformerSentence():
    def __init__(self, sentence_str, 
                 model=BertModel.from_pretrained('scibert-scivocab-uncased'), 
                 tokenizer=BertTokenizer.from_pretrained('scibert-scivocab-uncased')):
        
        self.raw_string = sentence_str
        self.model = model
        self.tokenizer = tokenizer
        self.summary = {}

        
    def write_summary(self, input_tokens=None, 
                      hidden_states=None, 
                      hidden_attentions=None,
                      print_tokens=True):
        
        if (input_tokens or hidden_states or hidden_attentions) is None:
            input_tokens, hidden_states, hidden_attentions = self.forward()
        
        # this replaces adds a "_{counter}" to the repreated tokens, so that 
        # they can be used uniquely as the keys for the embeddings dictionary
        input_tokens = TransformerSentence.make_unique(input_tokens)
        
        if print_tokens:
            print('Sentence Tokenization: ', input_tokens)
            
        # write summary into the object
        self.summary['input_tokens'] = input_tokens
        self.summary['states'] = hidden_states
        self.summary['attentions'] = hidden_attentions

        self.summary['token_embeddings'] = {input_token: hidden_states[:, i, :] 
                                            for i, input_token in enumerate(input_tokens)}
        
    def forward(self):
        encoded_inputs_dict = self.tokenizer.encode_plus(self.raw_string)
        input_ids = encoded_inputs_dict['input_ids']
        input_tensor = torch.tensor([input_ids])
        input_tokens = [self.tokenizer.decode(input_ids[j]).replace(' ', '') 
                        for j in range(len(input_ids))]
        
        final_attention, final_state, hidden_states_tup, hidden_attentions_tup = self.model(input_tensor)
        
        # stacking states and attentions along the first dimention (which corresponds to the batch when necessary)
        hidden_attentions = torch.cat(hidden_attentions_tup, dim=0) # 'layers', 'heads', 'queries', 'keys'
        hidden_states = torch.cat(hidden_states_tup, dim=0) # 'layers', 'tokens', 'embeddings'
        
        return input_tokens, hidden_states.detach(), hidden_attentions.detach()
    
    
    def attention_from_tokens(self, token1, token2, display=True):
        input_tokens = self.summary['input_tokens']
        
        if (token1 and token2) not in input_tokens:
            raise ValueError('One or both of the tokens introduced are not in the sentence!')
            
        idx1, idx2 = input_tokens.index(token1), input_tokens.index(token2)
        attention = self.summary['attentions'][:, :, idx1, idx2].numpy()
        if display:
            TransformerSentence.display_attention(attention, title=(token1, token2))
        return attention
    
    
    def attention_from_idx(self, i, j, display=True):
        attention = self.summary['attentions'][:, :, i, j].numpy()
        if display:
            TransformerSentence.display_attention(attention, title=f'Token idx: {(i, j)}')
        return attention
    
    def visualize_token_path(self, fit, 
                             tokens_to_follow=None, 
                             print_tokens=False, 
                             fig_axs=(None, None), 
                             figsize=(10, 10)):
        
        if tokens_to_follow is None:
            all_tokens = self.summary['input_tokens']
            regex = re.compile(r'^[a-zA-Z]')
            tokens_to_follow = [i for i in all_tokens if regex.search(i)]
            
        if print_tokens: print(tokens_to_follow)  
            
        colors = list(range(len(tokens_to_follow)))
        projections = []
        layer_depth = self.summary['states'].size()[0]
        
        for i in range(layer_depth):
            layer_embeddings = self.summary['states'][i, :, :]
            projection = fit.transform(layer_embeddings)
            projections.append(projection)

        data = np.stack(projections, axis=0)
        if None in fig_axs:
            fig, axs = plt.subplots(figsize=figsize)
        for token in tokens_to_follow:
            i = self.summary['input_tokens'].index(token)
            plt.plot(data[:,i,0], data[:,i,1], '-o', alpha=0.3)
            plt.annotate(s=token, xy=(data[0, i, 0], data[0, i, 1]))

        plt.show()
        
    def visualize_sentence_shape(self, fit, tokens_to_follow=None, 
                                 print_tokens=False, 
                                 fig_axs=(None, None), 
                                 figsize=(10, 10)):

        if tokens_to_follow is None:
            all_tokens = self.summary['input_tokens']
            regex = re.compile(r'^[a-zA-Z]')
            tokens_to_follow = [i for i in all_tokens if regex.search(i)]

        if print_tokens: print(tokens_to_follow)  

        colors = list(range(len(tokens_to_follow)))
        projections = []
        layer_depth = self.summary['states'].size()[0]
        
        # get list of indeces of the tokens to follow
        idxs = [self.summary['input_tokens'].index(token) for token in tokens_to_follow] 
        token_embeddings = self.summary['states'][-1, idxs, :]
        data = fit.transform(token_embeddings)
        
        if None in fig_axs:
            fig, axs = plt.subplots(figsize=figsize)
            
        plt.plot(data[:,0], data[:,1], '-o')
        for i, token in enumerate(tokens_to_follow):
            plt.annotate(s=token, xy=(data[i, 0], data[i, 1]))
        #plt.show()
    
    
    def save(self, name, path='.'):
        with open(os.path.join(path, name), 'wb') as file:
            pickle.dump(self, file)
    
    
    @staticmethod
    def visualize_embedding(embedding, title=None, vmax=None, vmin=None):
        if (vmax or vmin) is None:
            vmax = max(embedding)
            vmin = min(embedding)
            
        N = embedding.size()[0]
        h = math.ceil(math.sqrt(N))
        # N = a*b where abs(a-b) is minimum
        while (N % h != 0):
            h -= 1
        w = int(N / h)
        visualization = embedding.reshape((h, w)).numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(visualization, vmax=vmax, vmin=vmin, cmap='viridis')
        fig.colorbar(im)
        if title is not None:
            ax.set_title(title)
        plt.show()
    
    @staticmethod
    def display_attention(attention, title=None):
        fig, ax = plt.subplots()
        im = ax.imshow(attention, vmin=0., vmax=1., cmap='viridis')
        fig.colorbar(im)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('HEADS')
        ax.set_ylabel('LAYERS')
        plt.show()
    
    @staticmethod
    def load(name, path='.'):
        with open(os.path.join(path, name), 'rb') as file:
            SentenceObject = pickle.load(file)
        return SentenceObject
    
    @staticmethod
    def make_unique(L):
        unique_L = []
        for i, v in enumerate(L):
            totalcount = L.count(v)
            count = L[:i].count(v)
            unique_L.append(v + '_' + str(count+1) if totalcount > 1 else v)
        return unique_L
        
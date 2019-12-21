from custom.transformer_sentence import TransformerSentence
from typing import List
import itertools
import torch
import re

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5) # default similarity func.

class Chunker():
    def __init__(self, 
                 layer=-1, 
                 sim_function=cos, 
                 threshold=0.9, 
                 exclude_special_tokens=True, 
                 combinatorics='sequential'):

        self.layer = layer
        self.sim_function = sim_function
        self.threshold = threshold
        self.exclude_special_tokens = exclude_special_tokens

        if combinatorics not in ['sequential', 'all']: 
            raise ValueError("You must specify the combinatorics as 'sequencial' or 'all'!!")
        self.combinatorics = combinatorics
        

    def compact(self, sentence, threshold=None, layer=None):
        if threshold is not None: self.threshold = threshold
        if layer is not None: self.layer = layer

        assert type(sentence) == TransformerSentence #, "Input must be a TransformerSentence Object!"
        if not sentence.summary:
            sentence.write_summary()

        indices_to_compact = self.indices_to_compact_by_similarity_threshold(sentence)
        new_embeddings = self.compact_embeddings_avg(sentence, indices_to_compact)
        new_tokens = self.new_tokens(sentence, indices_to_compact)

        return new_embeddings, new_tokens


    def indices_to_compact_by_similarity_threshold(self, sentence) -> List[List]:
        
        sequence_embeddings = sentence.summary['states'][self.layer, :, :]
        
        # combinatorics= 'sequential', 'all'
        seq_length, embedding_size = sequence_embeddings.size() #make sure the input is proper size!!
        indices = list(range(seq_length))    
        
        # Combinations of indices that are group candidates
        if self.combinatorics == 'sequential':
            if self.exclude_special_tokens:
                idx_combinations = [indices[s:e] for s, e in itertools.combinations(range(1, len(indices)), 2)]
            else:
                idx_combinations = [indices[s:e] for s, e in itertools.combinations(range(len(indices)+1), 2)]
                
        elif self.combinatorics == 'all':
            idx_combinations = []
            for L in range(2, seq_length+1):
                combinations = list(itertools.combinations(indices, r=L))
                idx_combinations.extend(combinations)
        
        
        all_indices_to_compact = []
        for indices in idx_combinations:
            group_candidate = sequence_embeddings[indices, :]
            group_size = len(indices)
            center = torch.mean(group_candidate, dim=0)
            center = center.repeat(group_size, 1)
            # calculate all embeddings similarities w.r.t. the center of the group
            similarities = self.sim_function(center, group_candidate)
            worst_sim, _ = torch.min(similarities, dim=0)
            if worst_sim > self.threshold: all_indices_to_compact.append(indices)
                
        indices_to_compact = Chunker.remove_subsets(all_indices_to_compact)
        
        return indices_to_compact


    def compact_embeddings_avg(self, sentence, indices_to_compact) -> torch.Tensor:
        original_embeddings = sentence.summary['states'][self.layer, :, :]
        new_embeddings_list = []
        for indices in indices_to_compact:
            group = original_embeddings[indices, :]
            center = torch.mean(group, dim=0)
            new_embeddings_list.append(center)

        try:  
            new_embeddings = torch.stack(new_embeddings_list, dim=0)
        except:
            print("No chunks were found!")
            new_embeddings = None
    
        return new_embeddings


    def new_tokens(self, sentence, indices_to_compact) -> List[str]:
        sentence_tokens = sentence.summary['input_tokens']
        new_tokens = []
        for chunk_indices in indices_to_compact:
            if len(chunk_indices) > 1:
                tokens = [sentence_tokens[i] for i in chunk_indices]
                # remove everything after _ in tokens to match the words
                tokens = [re.sub(r'_(.*)', '', token) for token in tokens]
                joint_token = '_'.join(tokens).replace('_##', '')
                new_tokens.append(joint_token)
            else:
                
                token = sentence_tokens[chunk_indices[0]]
                new_tokens.append(token)

        return new_tokens


    @staticmethod
    def remove_subsets(L):
        filtered = filter(lambda f: not any(set(f) < set(g) for g in L), L)
        return list(filtered)

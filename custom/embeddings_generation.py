import torch

"""
The idea is to have a class that inicializes the representation generation mechanism.
Then, a forward pass gives a tensor (or batch of tensors) and the corresponding list of indices
and performs the generation.
"""

class EmbeddingGenerator():
    def __init__(self, pool_function=None):
        try:
            t = torch.Tensor([1, 2])
            _ = pool_function(t)
            self.pool_function = pool_function
        except:
            raise ValueError("The pool_function seems to not work!")
        

    def forward(self, input_tensors, indices):
        compact_representation = self.generate(input_tensors, indices)
        return compact_representation


    def generate(self, tensors_batch, indices_batch):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)]]

        compact_tensors_batch = self.initialize_padding_tensor_like(tensors_batch)
        # as all are zeros, this starts as an all false boolean mask
        mask = torch.zeros(tensors_batch.size(), dtype=torch.bool)

        for b, chunk_indices in enumerate(indices_batch):
            for i, idx_tuple in enumerate(chunk_indices):
                # for each group calculate the compacted with "pool_function" (which will eventually
                # be something more complex, not just max pooling)
                joint = self.pool_function(tensors_batch[b, idx_tuple, :])
                compact_tensors_batch[b, i, :] = joint
                mask[b, i, :] = True

        return compact_tensors_batch, mask


    def initialize_padding_tensor_like(self, tensor):
        # this function should be better, like initialize randomly from a distribution, because
        # the elements that are not overwritten by the originial tensors or pooling from them
        # will be padding
        init_tensor = torch.zeros_like(tensor)
        return init_tensor



def max_pooling(input_tensors, dim=0):
    # input tensors have shape (n, EmbDimension), 
    # this is problematic for now because this operation cannot be parallelized on GPU
    tensor, _ = torch.max(input_tensors, dim=0)
    return tensor

def avg_pooling(input_tensors, dim=0):
    # input tensors have shape (n, EmbDimension)
    # this is problematic for now because this operation cannot be parallelized on GPU
    avg_tensor = torch.mean(input_tensors, dim=0)
    return 


def filter_indices(indices_batch):
    
    for b, indices in enumerate(indices_batch):
        for i, idx in enumerate(indices):
            if len(idx) == 1:
                indices_batch[b].pop(i)

    return indices_batch

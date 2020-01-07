import torch
import torch.nn as nn
import torch.nn.functional as F

# this class is heavily based on the one implemented in the tutorial from pytorch 
# and forum https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sentset_size, num_layers, batch_size, bidirectional=True, dropout=0.):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_layers, dropout=dropout, batch_first=True, bias=False) 

        # The linear layer that maps from hidden state space to sentiment classification space
        self.hidden2sent = nn.Linear(hidden_dim * 2, sentset_size)
        self.hidden = self.init_hidden()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')


    def init_hidden(self):

        if self.bidirectional:
            directions = 2
        else:
            directions = 1

        # As of the documentation from nn.LSTM in pytorch, the input to the lstm cell is 
        # the input and a tuple of (h, c) hidden state and memory state. We initialize that
        # tuple with the proper shape: num_layers*directions, batch_size, hidden_dim. Don't worry
        # that the batch here is second, this is dealt with internally if the lstm is created with
        # batch_first=True
        shape = (self.num_layers * directions, self.batch_size, self.hidden_dim)
        return (torch.zeros(shape, requires_grad=True), torch.zeros(shape, requires_grad=True))


    def loss(self, predicted, target):
        return self.loss_fn(predicted, target)

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        # we use the last lstm output for classification
        sent_space = self.hidden2sent(lstm_out[:, -1, :])
        sent_scores = F.softmax(sent_space, dim=1)
        return sent_scores

import torch
from torch import nn


# Taken from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class FeatureMapper(nn.Module):
    def __init__(
        self, n_cells, input_size,
        lstm_hidden_size
    ):
        super(FeatureMapper, self).__init__()
        self.n_cells = n_cells

        self.fcs = []
        self.ats = []
        self.lstm_cells = []
        for i in range(n_cells):
            self.ats.append()
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(input_size, fc_output_size),
                    nn.ReLU()  # Do we need a ReLU here?
                )
            )
            self.lstm_cells.append(
                nn.LSTMCell(at_output_size, lstm_hidden_size))

        self.final_fcs = nn.Sequential(
            nn.Linear(lstm_hidden_size, final_fcs_hidden),
            nn.ReLU(),
            nn.Linear(final_fcs_hidden, output_size),
        )

    def forward(self, inputs):
        xs, (hx, cx) = inputs

        for i in range(self.n_cells):
            xs[i] = self.fcs[i](xs[i])
            xs[i] = self.ats[i](xs[i])

        hxs = []
        hxs[0] = hx

        # Connect LSTM cells
        for i in range(self.n_cells):
            hx_out, cx = self.lstm_cells[i](
                xs[i], (hxs[i], cx))
            hxs.append(hx_out)

        hxs = torch.cat(hxs, dim=0)

        return

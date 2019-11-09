import torch
from torch import nn


class FeatureMapper(nn.Module):
    def __init__(
        self, n_cells, input_size,
        lstm_hidden_size
    ):
        super(FeatureMapper, self).__init__()
        self.n_cells = n_cells

        fc_size = 256
        final_fcs_hidden = 256
        output_size = 20

        self.fcs = []
        self.lstm_cells = []
        for i in range(n_cells):
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(input_size, fc_size),
                    nn.ReLU()  # Do we need a ReLU here?
                )
            )
            self.lstm_cells.append(
                nn.LSTMCell(fc_size, lstm_hidden_size))

        self.final_fcs = nn.Sequential(
            nn.Linear(lstm_hidden_size * n_cells, final_fcs_hidden),
            nn.ReLU(),
            nn.Linear(final_fcs_hidden, output_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        xs, (hx, cx) = inputs

        for i in range(self.n_cells):
            xs[i] = self.fcs[i](xs[i])

        hxs = []
        hxs[0] = hx

        # Connect LSTM cells
        for i in range(self.n_cells):
            hx_out, cx = self.lstm_cells[i](
                xs[i], (hxs[i], cx))
            hxs.append(hx_out)

        hxs = torch.cat(hxs, dim=0)

        x = self.final_fcs(hxs)

        return x

import torch
import torch.nn                         as nn
import torch.optim
from torch.autograd                     import Variable
from core_qnn.quaternion_layers         import *
from utilities                          import normalize_quaternions


class QALLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def quaternion_conjugate(self, q):
        w, v = q[:, 0], q[:, 1:]
        return torch.cat((w.unsqueeze(-1), -v), dim=1)

    def quaternion_multiply(self, q1, q2):
        w1, v1 = q1[:, 0], q1[:, 1:]
        w2, v2 = q2[:, 0], q2[:, 1:]

        w = w1 * w2 - (v1 * v2).sum(dim=1)
        v = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + torch.cross(v1, v2)

        return torch.cat((w.unsqueeze(-1), v), dim=1)

    def forward(self, output: torch.tensor, expected: torch.tensor) -> torch.tensor:
        distance = self.quaternion_multiply(self.quaternion_conjugate(output), expected)
        w = distance[:, 0]
        angles_rad = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
        angles_rad = angles_rad**2
        return torch.mean(angles_rad)


class StackedQLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, n_layers, batch_first, device):
        super(StackedQLSTM, self).__init__()
        
        self.batch_first =  batch_first
        self.layers =       nn.ModuleList([QLSTM(feat_size, hidden_size, device) for _ in range(n_layers)])

    def forward(self, x):
        # QLSTM takes inputs of shape (seq_len, batch_size, feat_size)
        if self.batch_first:
            x = x.permute(1,0,2)

        for layer in self.layers:
            x = layer(x)

        if self.batch_first:
            x = x.permute(1,0,2)
        
        # Sentiment classification!
        x = x[:, -1, :]
        return x


class QLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, device):
        super(QLSTM, self).__init__()

        # Reading options:
        self.act =          nn.Tanh()
        self.act_gate =     nn.Sigmoid()
        self.input_dim =    feat_size
        self.hidden_dim =   hidden_size
        self.device =       device
        self.num_classes =  feat_size

        # Gates initialization
        self.wfx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim) # Forget
        self.ufh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False) # Forget

        self.wix = QuaternionLinearAutograd(self.input_dim, self.hidden_dim) # Input
        self.uih = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False) # Input

        self.wox = QuaternionLinearAutograd(self.input_dim, self.hidden_dim) # Output
        self.uoh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False) # Output

        self.wcx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim) # Cell
        self.uch = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False) # Cell

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):

        h_init = Variable(torch.zeros(x.shape[1],self. hidden_dim))
        h_init = h_init.to(self.device)
        x = x.to(self.device)

        # Feed-forward affine transformation (done in parallel)
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)

        # Processing time steps
        out = []
        c = h_init
        h = h_init

        for k in range(x.shape[0]):

            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))

            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)

            output = self.fco(h)
            out.append(output.unsqueeze(0))

        output = torch.cat(out,0)
        output = normalize_quaternions(output, dim=2)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()

        self.num_layers =   num_layers
        self.hidden_size =  hidden_size
        self.lstm =         nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first -> (batch_size, seq, input_size)
        self.fc =           nn.Linear(hidden_size, num_classes)
        self.device =       device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))     # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]                 # out: [Sentiment classification!]
        out = self.fc(out)
        out = normalize_quaternions(out, dim=1)
        return out
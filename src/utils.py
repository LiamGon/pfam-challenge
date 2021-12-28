import numpy as np # linear algebra
from scipy.stats import gaussian_kde

import torch
from prettytable import PrettyTable


class PDataset(torch.utils.data.Dataset):
    def __init__(self, max_len, vocab, seqs, labels):
        self.max_len = max_len
        self.vocab = vocab
        self.seqs = list(seqs)
        # self.tokenized_seqs = [self.tokenize_seq(seq) for seq in tqdm(seqs)]
        self.labels = torch.Tensor(list(labels)).to(torch.int64)

    def tokenize_seq(self, seq):
        t = torch.Tensor([self.vocab[key] for key in seq]).to(torch.int64)
        return torch.nn.functional.pad(
            t.unsqueeze(0),
            value=0,
            pad=(0, self.max_len - t.shape[0]),
        ).squeeze(0)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.tokenize_seq(self.seqs[idx]), self.labels[idx]


def count_parameters(model, print_table=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print("Total Trainable Params: {:,}".format(total_params))
    table.sortby = "Parameters"
    table.reversesort = True

    if print_table:
        print(table)
    return total_params


def get_dens(data, cov = 0.1):
    density = gaussian_kde(data)
    xs = np.linspace(0, np.max(data), 100)
    density.covariance_factor = lambda: cov
    density._compute_covariance()
    return xs, density(xs)

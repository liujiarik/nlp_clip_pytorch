import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, cutoffs, div_value=4.):
        super(AdaptiveEmbedding, self).__init__()
        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= num_embeddings) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and num_embeddings-1")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cutoffs = cutoffs
        self.div_value = div_value

        self.n_clusters = len(self.cutoffs) + 1
        self.edges = [0] + self.cutoffs + [num_embeddings]
        self.projections = nn.ModuleList()
        for i in range(self.n_clusters):
            hsz = int(self.embedding_dim // (self.div_value ** i))
            vsz = self.edges[i + 1] - self.edges[i]
            projection = nn.Sequential(
                nn.Embedding(vsz, hsz),
                nn.Linear(hsz, self.embedding_dim, bias=False)
            )

            self.projections.append(projection)

    def forward(self, emb_input: torch.Tensor):

        batch_size, seq_len = emb_input.size()
        emb_input = emb_input.view(-1)
        emb_output = emb_input.new_empty(batch_size * seq_len, self.embedding_dim).float()

        for i in range(self.n_clusters):
            low_idx = self.edges[i]
            high_idx = self.edges[i + 1]
            input_mask = (emb_input >= low_idx) & (emb_input < high_idx)
            row_indices = input_mask.nonzero().squeeze(dim=-1)
            if row_indices.numel() == 0:
                continue
            input_subset = emb_input.index_select(0, row_indices)
            input_subset = input_subset - low_idx
            cluster_output = self.projections[i](input_subset)
            emb_output.index_copy_(0, row_indices, cluster_output)

        return emb_output.view(batch_size, seq_len, -1)

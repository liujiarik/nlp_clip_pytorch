# Star-Transformer
An implementation of "Star-Transformer" in Pytorch 

example
```python
from strm_modeling import StarTransformerTokenClassifier, StarTransformerClassifier

import torch

v_size = 200
batch_size = 32
max_seq_len = 512

# make synthetic data, sentences in batch with word indexs
synthetic_seq_input = torch.randint(0, v_size, (batch_size, max_seq_len))

cycle_num = 2  # it's similar to layers_nums
hidden_size = 100
num_attention_heads = 5
attention_dropout_prob = 0.1
# 1. classification
model = StarTransformerClassifier(v_size, cycle_num, hidden_size, num_attention_heads, attention_dropout_prob,
                                  label_num=2)
o = model(synthetic_seq_input)
print(o.size())  # [32,2]

# 1. token classification & sequence labeling

model = StarTransformerTokenClassifier(v_size, cycle_num, hidden_size, num_attention_heads, attention_dropout_prob,
                                       label_num=4)
o = model(synthetic_seq_input)
print(o.size())  # [32,512,4]

```


# Requirement

Python 3.6 </br>
Pytorch 1.0 </br>

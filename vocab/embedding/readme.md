Download glove.42B.300d.zip from [here](https://nlp.stanford.edu/projects/glove/) and
unzip it into this directory. Note that this download might take some time as the file is
1.75 GB. Then run embeddings.ipynb and pca_project.ipynb.

You can use something like the below code to set the weights of a nn.Embedding layer where
'glove' is the PCA-projected, reduced vocabulary word embedding matrix.

```python
def set_embedding_weights(self, glove):
    for idx, token in enumerate(self.vocab.token2idx):
        weight = glove[token]
        self.embed.weight[idx].data = torch.from_numpy(weight)
```

The above code exists in decoder.py.

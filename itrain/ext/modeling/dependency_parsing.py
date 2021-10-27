"""
Code taken and modified from: https://github.com/Adapter-Hub/hgiyt.
Credits: P. Rust et al.
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.adapters.heads import PredictionHead


# Credit:
# Class taken from https://github.com/yzhangcs/biaffine-parser
class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        return s


class BiaffineParsingHead(PredictionHead):
    """
    Credit: G. Glavaš & I. Vulić
    Based on paper "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation"
    (https://arxiv.org/pdf/2008.06788.pdf)
    """

    def __init__(self, head_name, config, model):
        super().__init__(head_name)
        self.config = config
        self.model_config = model.config
        self.build(model)

    def build(self, model):
        self.biaffine_arcs = Biaffine(n_in=model.config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(
            n_in=model.config.hidden_size, n_out=self.config["num_labels"], bias_x=True, bias_y=True
        )

        self.dropout = nn.Dropout(model.config.hidden_dropout_prob)

        # imitates pooler
        # self.pooler_dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        # self.pooler_activation = nn.Tanh()

        self.loss_fn = CrossEntropyLoss()

        self.train(model.training)  # make sure training mode is consistent

    def forward(
        self, outputs, attention_mask, return_dict, word_starts=None, labels_arcs=None, labels_rels=None, **kwargs
    ):
        outs = self.dropout(outputs[0])
        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        # cls_output = self.pooler_activation(self.pooler_dense(outs[:, 0]))
        cls_output = outs[:, 0]
        word_outputs_heads = torch.cat([cls_output.unsqueeze(1), word_outputs_deps], dim=1)

        arc_preds = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_preds = arc_preds.squeeze()
        outputs = (arc_preds,)

        rel_preds = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_preds = rel_preds.permute(0, 2, 3, 1)
        outputs = (rel_preds,) + outputs

        loss = self._get_loss(arc_preds, rel_preds, labels_arcs, labels_rels, self.loss_fn)

        # TODO return_dict
        outputs = (loss,) + outputs

        if len(arc_preds.shape) == 2:
            return loss, rel_preds, arc_preds.unsqueeze(0)
        return outputs

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.model_config.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                vecs_range = subword_vecs[start:end]
                word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.model_config.hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to("cuda" if torch.cuda.is_available() else "cpu")

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.model_config.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss

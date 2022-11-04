import abc
import copy
from torch.autograd import Variable
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src
        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)
        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class STAGATEModule(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(STAGATEModule, self).__init__()
        [num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_features, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_features, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4


class StackClsModule(nn.Module, abc.ABC):
    def __init__(self, in_features, n_classes):
        super(StackClsModule, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes


class STAGATEClsModule(nn.Module):
    def __init__(self,
                 stagate: STAGATEModule,
                 stack_classifier: StackClsModule):
        super(STAGATEClsModule, self).__init__()
        self.stagate = copy.deepcopy(stagate)
        self.classifier = copy.deepcopy(stack_classifier)

    def forward(self, x, edge_index, mode="classifier"):
        z, x_recon = self.stagate(x, edge_index)
        z = torch.clone(z)
        if mode == "classifier":
            return z, self.classifier(z)
        elif mode == "reconstruction":
            return z, x_recon
        else:
            raise NotImplementedError

    def get_saliency_map(self, x, edge_index, target_index="max", save=None):
        """
        Get saliency map by backpropagation.
        :param x: input tensors
        :param edge_index: graph edge index
        :param target_index:  target index to compute final scores
        :param save:
        :return: gradients
        """
        x_var = Variable(x, requires_grad=True)
        _, output = self.forward(x_var, edge_index, mode="classifier")
        scores = output["last_layer"]
        if target_index == "max":
            target_score_indices = Variable(torch.argmax(scores, 1))
        elif isinstance(target_index, int):
            target_score_indices = Variable(torch.ones(scores.shape[0], dtype=torch.int64) * target_index)
        else:
            raise NotImplementedError
        target_scores = scores.gather(1, target_score_indices.view(-1, 1)).squeeze()
        loss = torch.sum(target_scores)
        loss.backward()
        gradients = x_var.grad.data
        if save is not None:
            torch.save(gradients, save)
        return gradients, scores


class StackMLPModule(StackClsModule):
    name = "StackMLP"

    def __init__(self, in_features, n_classes, hidden_dims=[30, 40, 30]):
        super(StackMLPModule, self).__init__(in_features, n_classes)
        self.classifier = nn.ModuleList()
        mlp_dims = [in_features] + hidden_dims + [n_classes]
        for ind in range(len(mlp_dims) - 1):
            self.classifier.append(nn.Linear(mlp_dims[ind], mlp_dims[ind + 1]))

    def forward(self, x):
        for layer in self.classifier:
            x = layer(x)
        score = F.softmax(x, dim=0)
        return {"last_layer": x, "score": score}

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and training objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).

    .. note::

        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.

    .. note::

        For an example of using :class:`LightGCN`, see `examples/lightgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        lightgcn.py>`_.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        agrregation_method : int, # 0 = sum, 1 = attention
        type_length: dict, # / user:n_user / item:n_item / test:n_test / tag:n_tag /
        alpha: Optional[Union[float, Tensor]] = None,
        
        **kwargs,
    ):
        super().__init__()
        self.agrregation_method = agrregation_method
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.type_length = type_length
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        ## linear for attn
        self.src_q_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim) 
        self.src_k_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim) ##dimesion reduce for generalization .
        self.src_v_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.src_mtn_calculator = torch.nn.MultiheadAttention(self.embedding_dim, 1)
        
        self.dst_q_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dst_k_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim) ##dimesion reduce for generalization .
        self.dst_v_converter = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dst_mtn_calculator = torch.nn.MultiheadAttention(self.embedding_dim, 1)
        
        self.user_item_embedding = Embedding(self.type_length["user"]+self.type_length["item"], embedding_dim)
        self.user_test_embedding = Embedding(self.type_length["user"]+self.type_length["test"], embedding_dim)
        self.user_tag_embedding = Embedding(self.type_length["user"]+self.type_length["tag"], embedding_dim)
        self.user_item_convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.user_test_convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.user_tag_convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.user_item_embedding.weight)
        torch.nn.init.xavier_uniform_(self.user_tag_embedding.weight)
        torch.nn.init.xavier_uniform_(self.user_test_embedding.weight)
        for conv in self.user_item_convs:
            conv.reset_parameters()
        for conv in self.user_test_convs:
            conv.reset_parameters()
        for conv in self.user_tag_convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        user_item_edge_index: Adj,
        user_test_edge_index: Adj,
        user_tag_edge_index: Adj,
        edge_weight: OptTensor = None, #none
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        user_item_x = self.user_item_embedding.weight
        user_item_out = user_item_x * self.alpha[0]
        user_tag_x = self.user_tag_embedding.weight
        user_tag_out = user_tag_x * self.alpha[0]
        user_test_x = self.user_test_embedding.weight
        user_test_out = user_test_x * self.alpha[0]
        
        for i in range(self.num_layers):
            user_item_x = self.user_item_convs[i](user_item_x, user_item_edge_index, edge_weight)
            user_item_out = user_item_out + user_item_x * self.alpha[i + 1]
        
        for i in range(self.num_layers):
            user_tag_x = self.user_tag_convs[i](user_tag_x, user_tag_edge_index, edge_weight)
            user_tag_out = user_tag_out + user_tag_x * self.alpha[i + 1]        

        for i in range(self.num_layers):
            user_test_x = self.user_test_convs[i](user_test_x, user_test_edge_index, edge_weight)
            user_test_out = user_test_out + user_test_x * self.alpha[i + 1]     
        
        return {"user_item_out": user_item_out, "user_tag_out": user_tag_out, "user_test_out": user_test_out}

    def forward(
        self,
        user_item_edge_index: Adj,
        user_test_edge_index: Adj,
        user_tag_edge_index: Adj,
        user_item_edge_label_index: OptTensor = None,
        user_test_edge_label_index: OptTensor = None,
        user_tag_edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
        """
        if user_item_edge_label_index is None:
            if is_sparse(user_item_edge_index):
                user_item_edge_label_index, _ = to_edge_index(user_item_edge_index)
                user_test_edge_label_index, _ = to_edge_index(user_test_edge_index)
                user_tag_edge_label_index, _ = to_edge_index(user_tag_edge_index)
            else:
                user_item_edge_label_index = user_item_edge_index
                user_test_edge_label_index = user_test_edge_index
                user_tag_edge_label_index = user_tag_edge_index

        out = self.get_embedding(user_item_edge_index, user_test_edge_index, user_tag_edge_index, edge_weight)
        
        user_item_out_src = out['user_item_out'][user_item_edge_label_index[0]]
        user_item_out_dst = out['user_item_out'][user_item_edge_label_index[1]]
        
        user_test_out_src = out['user_test_out'][user_test_edge_label_index[0]]
        user_test_out_dst = out['user_test_out'][user_test_edge_label_index[1]]

        user_tag_out_src = out['user_tag_out'][user_tag_edge_label_index[0]]
        user_tag_out_dst = out['user_tag_out'][user_tag_edge_label_index[1]]
        
        if self.agrregation_method == 0: #sum
            result_src = user_item_out_src + user_test_out_src + user_tag_out_src #user embdings (batch, # of edge, hd)
            result_dst = user_item_out_dst + user_test_out_dst + user_tag_out_dst # item, test, tag embdings (batch, # of edge, hd)
            result_src *= (1/3)
            result_dst *= (1/3)
        elif self.agrregation_method == 1:
            stacked_src = torch.stack((user_item_out_src, user_test_out_src, user_tag_out_src),dim=0)
            stacked_dst = torch.stack((user_item_out_dst, user_test_out_dst, user_tag_out_dst),dim=0)
            
            src_q = self.src_q_converter(stacked_src)
            src_k = self.src_k_converter(stacked_src)
            src_v = self.src_v_converter(stacked_src)
            dst_q = self.dst_q_converter(stacked_dst)
            dst_k = self.dst_k_converter(stacked_dst)
            dst_v = self.dst_v_converter(stacked_dst)
            
            src_mtn, _ = self.src_mtn_calculator(src_q, src_k, src_v)
            dst_mtn, _ = self.dst_mtn_calculator(dst_q, dst_k, dst_v)
            
            result_src = torch.sum(src_mtn,dim=0)
            result_dst = torch.sum(dst_mtn,dim=0)
        else:
            raise("no_aggregation_method")
        
        '''
        out_ele_user_item = user_item_out_src*user_item_out_dst
        out_ele_user_item = out_ele_user_item.sum(dim=-1)
        out_ele_user_test = user_test_out_src*user_test_out_dst
        out_ele_user_test = out_ele_user_test.sum(dim=-1)
        out_ele_user_tag = user_tag_out_src*user_tag_out_dst
        out_ele_user_tag = out_ele_user_tag.sum(dim=-1)
        '''
        return (result_src*result_dst).sum(dim=-1)

    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
            prob (bool, optional): Whether probabilities should be returned.
                (default: :obj:`False`)
        """
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (torch.Tensor): The predictions.
            edge_label (torch.Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

  
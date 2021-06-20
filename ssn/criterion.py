import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


class Criterion(nn.Module):
    """ This class computes the loss for SSN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth entities and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and boundary)
    """
    def __init__(self, entity_types, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            entity_types: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between ground-truth and prediction
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            losses: list of all the losses to be applied
        """
        super().__init__()
        self.entity_types = entity_types
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses


    def loss_class(self, outputs, targets, indices, num_spans):
        """Classification loss (cross entropy loss)
        targets dicts must contain the key "labels"
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        targets["labels"] = targets["labels"].split(targets["sizes"], dim=-1)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        empty_weight = torch.ones(src_logits.size(-1), device=src_logits.device)
        empty_weight[0] = num_spans / (src_logits.size(0) * src_logits.size(1) - num_spans)
        loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)

        losses = {'loss_class': loss_class}

        return losses


    def loss_boundary(self, outputs, targets, indices, num_spans):
        """Boundary loss (negative log likelihood loss)
        targets dicts must contain the key "spans"
        """
        assert 'pred_boundaries' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_spans_left = outputs['pred_boundaries'][0][idx]
        src_spans_right = outputs['pred_boundaries'][1][idx]

        targets["spans"] = targets["spans"].split(targets["sizes"], dim=0)
        target_spans = torch.cat([t[i] for t, (_, i) in zip(targets["spans"], indices)], dim=0)

        src_spans_left_logp = torch.log(1e-25 + src_spans_left)
        src_spans_right_logp = torch.log(1e-25 + src_spans_right)

        left_nll_loss = F.nll_loss(src_spans_left_logp, target_spans[:, 0], reduction='none')
        right_nll_loss = F.nll_loss(src_spans_right_logp, target_spans[:, 1], reduction='none')

        loss_boundary = left_nll_loss + right_nll_loss

        losses = {}
        losses['loss_boundary'] = loss_boundary.sum() / num_spans

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'class': self.loss_class,
            'boundary': self.loss_boundary
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target spans, for normalization purposes
        num_spans = sum(targets["sizes"])
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans, **kwargs))

        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_span: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_span: This is the relative weight of the boundary error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        assert cost_class != 0 or cost_span != 0, "all costs can't be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, entity_types] with the classification logits
                 "pred_boundaries": Tensor of dim [batch_size, num_queries, 2] with the predicted boundaries

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_spans] 
                 "spans": Tensor of dim [num_target_spans, 2] containing the target spans

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1)

            entity_left = outputs["pred_boundaries"][0].flatten(0, 1)
            entity_right = outputs["pred_boundaries"][1].flatten(0, 1)

            tgt_ids = targets["labels"]
            tgt_spans = targets["spans"]

            cost_class = -out_prob[:, tgt_ids]
            cost_span = -(entity_left[:, tgt_spans[:, 0]] + entity_right[:, tgt_spans[:, 1]])

            # Final cost matrix
            C = self.cost_span * cost_span + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()
            
            # Hungarian match
            sizes = targets["sizes"]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

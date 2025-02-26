import torch
from torch import nn, autograd
from torch.nn import functional as F


class ExemplarMemory(autograd.Function):

    @staticmethod
    #@amp.custom_fwd
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    #@amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


class ProxyMemory(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2, negK=50):
        super(ProxyMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = torch.tensor(momentum).to(torch.device('cuda'))
        self.proxy_memory = None
        self.all_proxy_label = ''   # torch tensor, cluster label of each proxy in proxy_memory
        self.img_proxy_index = ''  # torch tensor, from unique image index to proxy index
        self.negK = negK
        self.all_proxy_mask = ''


    def forward(self, features, index_labels, detach_feat=False):
        # Note: features: shape= (2*B, C), index_labels shape= (B,)

        proxy_index_label = self.img_proxy_index[index_labels]
        batch_pseudo_label = self.all_proxy_label[proxy_index_label]
        
        if detach_feat:
            ori_scores = torch.mm(features, self.proxy_memory.detach().clone().t())
        else:
            ori_scores = ExemplarMemory.apply(features, proxy_index_label, self.proxy_memory, self.momentum)
        scores = ori_scores / self.temp

        # when using weighted sampler, compute memory contrast loss only on valid samples
        val_inds = torch.nonzero(proxy_index_label >= 0).squeeze(-1)
        assert(len(val_inds) == len(scores))
        #scores = scores[val_inds]
        #batch_pseudo_label = batch_pseudo_label[val_inds]

        temp_score = scores.detach().clone()
        temp_k = min(self.negK, temp_score.size(1))

        loss = 0
        if len(val_inds)>0:
            memory_loss = 0
            for i in range(len(scores)):
                pos_ind = torch.nonzero(self.all_proxy_label==batch_pseudo_label[i]).squeeze(-1)
                temp_score[i, pos_ind] = 1000  # mark the positives
                _, top_inds = torch.topk(temp_score[i], temp_k)
                sel_score = scores[i, top_inds]
                target_label = torch.zeros(sel_score.shape, dtype=scores.dtype).cuda()
                target_label[0:len(pos_ind)] = 1.0/len(pos_ind)
                memory_loss += -(F.log_softmax(sel_score.unsqueeze(0), dim=1) * target_label.unsqueeze(0)).sum()
            loss += memory_loss/len(scores)                

        return loss



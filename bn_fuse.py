import torch


def fuse_one(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    out_ch = conv_w.shape[0]
    scale = bn_w / torch.sqrt(bn_var + eps)
    shift = bn_b - bn_mean * scale
    W_fused = conv_w * scale.view(out_ch, 1, 1, 1)
    b_fused = shift
    return W_fused, b_fused


def fuse_backbone(pretrained_conv, bn_buffers):
    fused_conv = {}
    fused_bias = {}

    for name, conv_w in pretrained_conv.items():
        if name not in bn_buffers:
            fused_conv[name] = conv_w
            fused_bias[name] = torch.zeros(conv_w.shape[0])
            continue

        bn = bn_buffers[name]
        bn_w = bn["bn_w"]
        bn_b = bn["bn_b"]
        bn_mean = bn["bn_mean"]
        bn_var = bn["bn_var"]
        bn_eps = bn.get("bn_eps", torch.tensor(1e-5))
        if isinstance(bn_eps, torch.Tensor):
            bn_eps = bn_eps.item()

        W_f, b_f = fuse_one(conv_w, bn_w, bn_b, bn_mean, bn_var, bn_eps)
        fused_conv[name] = W_f
        fused_bias[name] = b_f

    return fused_conv, fused_bias

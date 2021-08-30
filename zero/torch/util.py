from torch.autograd import Function


def instance(instance, *kargs, **kwargs):
    return instance(*kargs, **kwargs) if type(instance).__name__ in ('type', 'function') else instance


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        λ = ctx.λ
        λ = grads.new_tensor(λ)
        dx = -λ * grads
        return dx, None

from torch.optim.lr_scheduler import LambdaLR


class LR:
    def __init__(self, d_model, warmup_step):
        self.d_model = d_model
        self.warmup_step = warmup_step

    def __call__(self, step):
        step = max(step, 1)
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_step ** -1.5)


def set_up_optimizer(model, optimizer, d_model, warmup_step):
    # the learning rate is set to 1 because we use LambdaLR to adjust the learning rate
    optim = optimizer([param for param in model.parameters() if param.requires_grad], 1)
    lr_scheduler = LambdaLR(optim, LR(d_model, warmup_step))
    return optim, lr_scheduler


if __name__ == "__main__":
    lr = LR(512, 4000)
    for i in range(100000):
        print("{}\n".format(lr(i)))

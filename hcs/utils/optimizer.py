import torch.optim as optim


def create_optimizer(parameters, optim_type='lbfgs',
                     lr=1e-3,
                     maxiters=20,
                     **kwargs):
    ''' Creates the optimizer
    '''
    if optim_type == 'adam':
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2),
                          weight_decay=weight_decay)
    elif optim_type == 'lbfgs':
        return optim.LBFGS(parameters, lr=lr, max_iter=maxiters)
    elif optim_type == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Optimizer {} not supported!'.format(optim_type))

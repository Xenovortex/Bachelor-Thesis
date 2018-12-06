from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re


def get_mnist_coder(mask_size=[28, 28]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of MNIST images
    :return:
    """

    img_dims = [1, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv1')
    conv2 = fr.Node([conv1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv2')
    conv3 = fr.Node([conv2.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (784, )}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'F_args':{}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


def get_mnist_conv(mask_size=[28, 28]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of MNIST images
    :return:
    """

    img_dims = [1, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': False}, 'clamp': 1}, name='conv1')
    conv2 = fr.Node([conv1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args':{'channels_hidden': 128, 'batch_norm': False}, 'clamp': 1}, name='conv2')
    conv3 = fr.Node([conv2.out0], la.rev_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': False}}, name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (784, )}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'F_args':{}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


def get_mnist_batchnorm(mask_size=[28, 28]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of MNIST images
    :return:
    """

    img_dims = [1, mask_size[0], mask_size[1]]


    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv1 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args':{'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv1')
    conv2 = fr.Node([conv1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv2')
    conv3 = fr.Node([conv2.out0], la.rev_layer, {'F_class': fu.F_conv, 'F_args':{'channels_hidden': 128, 'batch_norm': True}}, name='conv3')

    r2 = fr.Node([conv3.out0], re.reshape_layer, {'target_dim': (784, )}, name='r2')

    fc = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'F_args':{}, 'clamp': 1}, name='fc')

    r3 = fr.Node([fc.out0], re.reshape_layer, {'target_dim': (4, 14, 14)}, name='r3')

    r4 = fr.Node([r3.out0], re.haar_restore_layer, {}, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv1, conv2, conv3, r1, r2, r3, r4, fc]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


def get_cifar_reshape(mask_size=[32, 32]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of CIFAR images
    :return: autoencoder
    """

    img_dims = [3, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.reshape_layer, {'target_dim': (12, 16, 16)}, name='r1')

    conv11 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv11')
    conv12 = fr.Node([conv11.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv12')
    conv13 = fr.Node([conv12.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv13')

    conv14 = fr.Node([conv13.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv14')
    conv15 = fr.Node([conv14.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv15')
    conv16 = fr.Node([conv15.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv16')

    r2 = fr.Node([conv16.out0], re.reshape_layer, {'target_dim': (48, 8, 8)}, name='r2')

    conv21 = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv21')
    conv22 = fr.Node([conv21.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv22')
    conv23 = fr.Node([conv22.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv23')

    r3 = fr.Node([conv23.out0], re.reshape_layer, {'target_dim': (3072, )}, name='r3')

    fc1 = fr.Node([r3.out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'clamp': 1}, name='fc1')

    r4 = fr.Node([fc1.out0], re.reshape_layer, {'target_dim': (3, 32, 32) }, name='r4')

    outp = fr.OutputNode([r4.out0], name='output')

    nodes = [inp, outp, conv11, conv12, conv13, conv14, conv15, conv16, conv21, conv22, conv23, r1, r2, r3, r4, fc1]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder

'''
       NOT WORKING AT THE MOMENT

       img_dims = [3, mask_size[0], mask_size[1]]

       nodes = []

       nodes.append(fr.InputNode(*img_dims, name='input'))

       nodes.append(fr.Node([nodes[-1].out0], re.reshape_layer, {'target_dim': (12, 16, 16)}, name='r1'))

       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv11'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv12'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv13'))

       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv21'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv22'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv23'))

       nodes.append(fr.Node([nodes[-1].out0], re.reshape_layer, {'target_dim': (48, 8, 8)}, name='r2'))

       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv31'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv32'))
       nodes.append(fr.Node([nodes[-1].out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv33'))

       nodes.append(fr.Node([nodes[-1].out0], re.reshape_layer, {'target_dim': (3072, )}, name='r3'))

       nodes.append(fr.Node([nodes[-1].out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'clamp': 1}, name='fc1'))

       nodes.append(fr.Node([nodes[-1].out0], re.reshape_layer, {'target_dim': (3, 32, 32)}, name='r4'))

       nodes.append(fr.OutputNode([nodes[-1].out0], name='output'))

       coder = fr.ReversibleGraphNet(nodes, 0, -1)

       return coder

       '''


def get_cifar_haar(mask_size=[32, 32]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of CIFAR images
    :return: autoencoder
    """

    img_dims = [3, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv11 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv11')
    conv12 = fr.Node([conv11.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv12')
    conv13 = fr.Node([conv12.out0], la.rev_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}}, name='conv13')

    conv14 = fr.Node([conv13.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv14')
    conv15 = fr.Node([conv14.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv15')
    conv16 = fr.Node([conv15.out0], la.rev_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}}, name='conv16')

    r2 = fr.Node([conv16.out0], re.haar_multiplex_layer, {}, name='r2')

    conv21 = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv21')
    conv22 = fr.Node([conv21.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}, 'clamp': 1}, name='conv22')
    conv23 = fr.Node([conv22.out0], la.rev_layer, {'F_class': fu.F_conv, 'F_args': {'channels_hidden': 128, 'batch_norm': True}}, name='conv23')

    r3 = fr.Node([conv23.out0], re.reshape_layer, {'target_dim': (3072,)}, name='r3')

    fc1 = fr.Node([r3.out0], la.rev_multiplicative_layer, {'F_class': fu.F_fully_connected, 'clamp': 1}, name='fc1')

    r4 = fr.Node([fc1.out0], re.reshape_layer, {'target_dim': (48, 8, 8)}, name='r4')

    r5 = fr.Node([r4.out0], re.haar_restore_layer, {}, name='r5')

    r6 = fr.Node([r5.out0], re.haar_restore_layer, {}, name='r6')

    outp = fr.OutputNode([r6.out0], name='output')

    nodes = [inp, outp, conv11, conv12, conv13, conv14, conv15, conv16, conv21, conv22, conv23, r1, r2, r3, r4, r5, r6, fc1]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


def get_cifar_small(mask_size=[32, 32]):
    """
    Return an autoencoder.

    :param mask_size: size of the input. Default: Size of CIFAR images
    :return: autoencoder
    """

    img_dims = [3, mask_size[0], mask_size[1]]

    inp = fr.InputNode(*img_dims, name='input')

    r1 = fr.Node([inp.out0], re.haar_multiplex_layer, {}, name='r1')

    conv11 = fr.Node([r1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv11')
    conv12 = fr.Node([conv11.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv12')
    conv13 = fr.Node([conv12.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv13')

    conv14 = fr.Node([conv13.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv14')
    conv15 = fr.Node([conv14.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv15')
    conv16 = fr.Node([conv15.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv16')

    r2 = fr.Node([conv16.out0], re.haar_multiplex_layer, {}, name='r2')

    conv21 = fr.Node([r2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv21')
    conv22 = fr.Node([conv21.out0], la.rev_multiplicative_layer, {'F_class': fu.F_conv, 'clamp': 1}, name='conv22')
    conv23 = fr.Node([conv22.out0], la.rev_layer, {'F_class': fu.F_conv}, name='conv23')

    r3 = fr.Node([conv23.out0], re.reshape_layer, {'target_dim': (3072,)}, name='r3')

    fc1 = fr.Node([r3.out0], la.rev_multiplicative_layer, {'F_class': fu.F_small_connected, 'clamp': 1}, name='fc1')
    fc2 = fr.Node([fc1.out0], la.rev_multiplicative_layer, {'F_class': fu.F_small_connected, 'clamp': 1}, name='fc2')
    fc3 = fr.Node([fc2.out0], la.rev_multiplicative_layer, {'F_class': fu.F_small_connected, 'clamp': 1}, name='fc3')

    r4 = fr.Node([fc3.out0], re.reshape_layer, {'target_dim': (48, 8, 8)}, name='r4')

    r5 = fr.Node([r4.out0], re.haar_restore_layer, {}, name='r5')

    r6 = fr.Node([r5.out0], re.haar_restore_layer, {}, name='r6')

    outp = fr.OutputNode([r6.out0], name='output')

    nodes = [inp, outp, conv11, conv12, conv13, conv14, conv15, conv16, conv21, conv22, conv23, r1, r2, r3, r4, r5, r6, fc1, fc2, fc3]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder

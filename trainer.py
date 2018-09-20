# import os
import res
import time
import utils
import torch
import random
import numpy as np

import Vanilla

from torch                     \
    import Tensor
from torch.multiprocessing       \
    import Pool
thisPool = Pool

from Vanilla                         \
    import custom_entropy             \
    as loss_fn1
from Vanilla                            \
    import custom_distance               \
    as loss_fn2
loss_fn1 = loss_fn2 # hybrid-loss offline.

from Vanilla                                \
    import update_model_rmsprop              \
    as optimize_model


layer_sizes = [5, 8, 7] # [8, 12, 10] # [2, 4, 5] # [3, 5, 4]


epochs = 20
learning_rate = 0.001

data_size = 20_000 ; batch_size = 500
data_path = 'samples.pkl'


rms_alpha = 0.9

rmsadv_alpha_moment = 0.2
rmsadv_beta_moment = 0.8
rmsadv_alpha_accugrad = 0.9

adam_alpha_moment = 0.9
adam_alpha_accugrad = 0.999


drop_in = 0.0
drop_mid = 0.0
drop_out = 0.0

write_loss_to_txt = True



def train_rms(model, accu_grads, data, num_epochs=1, display_details=False):

    num_samples = len(data)
    num_batches = int(num_samples / batch_size)
    num_workers = torch.multiprocessing.cpu_count()

    if display_details:
        print('\n'
              f'\/ Training Started : Rms \/\n'
              f'Sample Amount: {num_samples}\n'
              f'Batch Size: {batch_size}\n'
              f'Workers: {num_workers}\n'
              f'Learning Rate: {learning_rate}\n'
              f'Epochs: {num_epochs}\n')

    losses = []

    for epoch in range(num_epochs):

        start_t = time.time()

        epoch_loss = np.zeros(Vanilla.hm_outs)

        random.shuffle(data)

        for batch in range(num_batches):

            # create batch

            batch_loss = np.zeros_like(epoch_loss)

            batch_ptr = batch * batch_size
            batch_end_ptr = (batch+1) * batch_size
            batch = data[batch_ptr:batch_end_ptr]

            with thisPool(num_workers) as pool:

                # create procs

                results = pool.map_async(process_fn, [[model.copy(), batch[_]] for _ in range(batch_size)])

                pool.close()

                # retrieve procs

                pool.join()

                for result in results.get():
                    loss, grads = result

                    Vanilla.apply_grads(model,grads)
                    batch_loss -= loss

                # handle

                epoch_loss += batch_loss

            # Vanilla.disp_grads(model)

            optimize_model(model, accu_grads, batch_size=batch_size, lr=learning_rate, alpha=rms_alpha)

            # Vanilla.disp_params(model)

        losses.append(epoch_loss)

        if display_details:
            if write_loss_to_txt: res.write_loss(epoch_loss, as_txt=True, epoch_nr=epoch)
            else: res.write_loss(epoch_loss)
            print(f'epoch {epoch+1} / {num_epochs} completed. PET: {round((time.time() - start_t),3)}')

        # res.save_model(model, epoch, asText=True)

    return model, accu_grads, losses


def process_fn(fn_input):

    model, data = fn_input
    x_vocab, x_oct, x_dur, x_vol, y_vocab, y_oct, y_dur, y_vol = data
    generative_length = len(y_vocab)

    inp = [Tensor(e) for e in [x_vocab, x_oct, x_dur, x_vol]]
    trg = [Tensor(e) for e in [y_vocab, y_oct, y_dur, y_vol]]

    response = Vanilla.forward_prop(model, inp, gen_iterations=generative_length)

    resp0, resp1, resp2, resp3 = [], [], [], []
    for resp_t in response:
        resp0.append(resp_t[0])    # response[:,0]
        resp1.append(resp_t[1])    # response[:,1]
        resp2.append(resp_t[2])    # response[:,2]
        resp3.append(resp_t[3])    # response[:,3]

    loss_nodes = [
        loss_fn1(resp0, trg[0]),
        loss_fn2(resp1, trg[1]),
        loss_fn2(resp2, trg[2]),
        loss_fn2(resp3, trg[3])]

    Vanilla.update_gradients(loss_nodes)

    loss = [float(sum(e)) for e in [loss_nodes[0], loss_nodes[1], loss_nodes[2], loss_nodes[3]]]
    # loss = [float(sum(node)) if _ == 0
    #         else -float(sum(node))
    #         for _, node in enumerate([loss_nodes[0], loss_nodes[1], loss_nodes[2], loss_nodes[3]])]

    grads = Vanilla.return_grads(model)

    return loss, grads



    #   Dev Purposes   #



def floyd_out(str):
    with open('/output/progress.txt', 'a+') as f:
        f.write(str)

def floyd_out_params(weights):
    names, params = weights
    with open('/output/param_data.txt', 'a+') as f:
        for i in range(len(params)):
            name = names[i]
            param = params[i]
            f.write(str(name) + ' ' + str(param) + ' ' + '\n')


def init_accugrads(model):
    accu_grads = []
    for layer in model:
        layer_accus = []
        for _ in layer.keys():
            layer_accus.append(0)
        accu_grads.append(layer_accus)
    return accu_grads

def save_accugrads(accu_grads, model_id=None):
    model_id = '' if model_id is None else str(model_id)
    res.pickle_save(accu_grads, 'model' + model_id + '_accugrads.pkl')

def load_accugrads(model, model_id=None):
    model_id = '' if model_id is None else str(model_id)
    try:
        accu_grads = res.pickle_load('model' + model_id + '_accugrads.pkl')
        print('> accugrads.pkl loaded.')
    except:
        print('> accugrads.pkl not found.')
        accu_grads = init_accugrads(model)
    return accu_grads


def init_moments(model):
    moments = []
    for layer in model:
        layer_moments = []
        for _ in layer.keys():
            layer_moments.append(0)
        moments.append(layer_moments)
    return moments

def save_moments(moments, model_id=None):
    model_id = '' if model_id is None else str(model_id)
    res.pickle_save(moments, 'model' + model_id + '_moments.pkl')

def load_moments(model, model_id=None):
    model_id = '' if model_id is None else str(model_id)
    try:
        moments = res.pickle_load('model' + model_id + '_moments.pkl')
        print('> moments.pkl loaded.')
    except:
        print('> moments.pkl not found.')
        moments = init_moments(model)
    return moments



    #   Alternative Trainers     #



def train_rmsadv(model, accu_grads, moments, data, epoch_nr=None, display_details=False):

    num_samples = len(data)
    num_batches = int(num_samples / batch_size)
    num_workers = torch.multiprocessing.cpu_count()

    if display_details:
        print('\n'
              f'\/ Training Started : Rms_adv \/\n'
              f'Sample Amount: {num_samples}\n'
              f'Batch Size: {batch_size}\n'
              f'Workers: {num_workers}\n'
              f'Learning Rate: {learning_rate}\n'
              f'Epochs: {num_epochs}\n')

    losses = []

    for epoch in range(epochs):

        start_t = time.time()

        epoch_loss = np.zeros(Vanilla.hm_outs)

        random.shuffle(data)

        for batch in range(num_batches):

            # create batch

            batch_loss = np.zeros_like(epoch_loss)

            batch_ptr = batch * batch_size
            batch_end_ptr = (batch+1) * batch_size
            batch = np.array(data[batch_ptr:batch_end_ptr])

            # proto = model.copy()
            # utils.nesterov_step1(proto, moments)

            with thisPool(num_workers) as pool:

                # create procs

                results = pool.map_async(process_fn_alt, [[model.copy(), batch[_]] for _ in range(batch_size)])

                pool.close()

                # retrieve procs

                pool.join()

                for result in results.get():
                    loss, grads = result

                    Vanilla.apply_grads(model,grads)
                    batch_loss -= loss

                # handle

                epoch_loss += batch_loss

            utils.nesterov_step2_adaptive(model, accu_grads, moments, batch_size=batch_size, lr=learning_rate, alpha_moments=rmsadv_alpha_moment, beta_moments=rmsadv_beta_moment, alpha_accugrads=rmsadv_alpha_accugrad)

        losses.append(epoch_loss)

        if display_details:
            if write_loss_to_txt: res.write_loss(epoch_loss, as_txt=True, epoch_nr=epoch)
            else: res.write_loss(epoch_loss)
            print(f'epoch {epoch+1} / {num_epochs} completed. PET: {round((time.time() - start_t),3)}')

    return model, accu_grads, moments, losses


def train_adam(model, accu_grads, moments, data, epoch_nr, display_details=False):

    num_samples = len(data)
    num_batches = int(num_samples / batch_size)
    num_workers = torch.multiprocessing.cpu_count()

    if display_details:
        print('\n'
              f'\/ Training Started : Adam \/\n'
              f'Sample Amount: {num_samples}\n'
              f'Batch Size: {batch_size}\n'
              f'Workers: {num_workers}\n'
              f'Learning Rate: {learning_rate}\n'
              f'Epochs: {num_epochs}\n')

    losses = []

    for epoch in range(epochs):

        start_t = time.time()

        epoch_loss = np.zeros(Vanilla.hm_outs)

        random.shuffle(data)

        for batch in range(num_batches):

            # create batch

            batch_loss = np.zeros_like(epoch_loss)

            batch_ptr = batch * batch_size
            batch_end_ptr = (batch+1) * batch_size
            batch = np.array(data[batch_ptr:batch_end_ptr])

            with thisPool(num_workers) as pool:

                # create procs

                results = pool.map_async(process_fn_alt, [[model.copy(), batch[_]] for _ in range(batch_size)])

                pool.close()

                # retrieve procs

                pool.join()

                for result in results.get():
                    loss, grads = result

                    Vanilla.apply_grads(model,grads)
                    batch_loss -= loss

                # handle

                epoch_loss += batch_loss

            utils.update_model_adam(model, accu_grads, moments, epoch_nr, batch_size=batch_size, lr=learning_rate, alpha_moments=adam_alpha_moment, alpha_accugrads=adam_alpha_accugrad)

        losses.append(epoch_loss)

        if display_details:
            if write_loss_to_txt: res.write_loss(epoch_loss, as_txt=True, epoch_nr=epoch)
            else: res.write_loss(epoch_loss)
            print(f'epoch {epoch+1} / {num_epochs} completed. PET: {round((time.time() - start_t),3)}')

    return model, accu_grads, moments, losses



def process_fn_alt(fn_input):

    model, data = fn_input
    x_vocab, x_oct, x_dur, x_vol, y_vocab, y_oct, y_dur, y_vol = data
    generative_length = len(y_vocab)

    inp = [Tensor(e) for e in [x_vocab, x_oct, x_dur, x_vol]]
    trg = [Tensor(e) for e in [y_vocab, y_oct, y_dur, y_vol]]

    response = utils.forward_prop_train(model, inp, gen_iterations=generative_length, drop_in=drop_in, drop_mid=drop_mid, drop_out=drop_out)

    resp0, resp1, resp2, resp3 = [], [], [], []
    for resp_t in response:
        resp0.append(resp_t[0])    # response[:,0]
        resp1.append(resp_t[1])    # response[:,1]
        resp2.append(resp_t[2])    # response[:,2]
        resp3.append(resp_t[3])    # response[:,3]

    loss_nodes = np.array([
        loss_fn1(resp0, trg[0]),
        loss_fn2(resp1, trg[1]),
        loss_fn2(resp2, trg[2]),
        loss_fn2(resp3, trg[3])])

    # loss = [float(sum(e)) for e in [loss_nodes[0], loss_nodes[1], loss_nodes[2], loss_nodes[3]]]
    loss = []
    for _, node in enumerate(loss_nodes):
        element = float(sum(node))
        if _ == 0:
            loss.append(element)
        else:
            loss.append(-element)

    Vanilla.update_gradients(loss_nodes)
    model_grads = Vanilla.return_grads(model)

    return loss, model_grads






if __name__ == '__main__':

    torch.set_default_tensor_type('torch.FloatTensor')

    data = res.load_data(data_path,data_size)
    IOdims = res.vocab_size

    if write_loss_to_txt:
        res.initialize_loss_txt()

    # # here is a sample datapoint (X & Y)..
    # print('X:')
    # for thing in data[0][0:4]: print(thing)
    # print('Y:')
    # for thing in data[0][4:]: print(thing)

    model = res.load_model()
    if model is None: model = Vanilla.create_model(IOdims,layer_sizes,IOdims)
    accu_grads = load_accugrads(model)

    model, accu_grads, losses = train_rms(model, accu_grads, data, num_epochs=epochs, display_details=True)
    res.save_model(model)
    save_accugrads(accu_grads)

import Vanilla

import torch
from torch import Tensor
import torch.nn.functional \
    as F

import res

import os
import glob
import random
import numpy as np


hm_ins = Vanilla.hm_ins
hm_outs = Vanilla.hm_outs
is_stop_cond = Vanilla.stop_cond

MAX_PROP_TIME = 20

grad_save_time = 15


    # Forward-Prop_math


def prop_func_alt(model, sequence_t, context_t, out_context_t, drop_in, drop_mid, drop_out):
    t_states = []

    # GRU Input Layer

    remember = F.sigmoid(
        model[0]['wr'] *
        (torch.matmul(model[0]['vr'], sequence_t) +
         torch.matmul(model[0]['ur'], context_t[0])
         )
        + model[0]['br']
    )

    attention = F.sigmoid(
        model[0]['wa'] *
        (torch.matmul(model[0]['va'], sequence_t) +
         torch.matmul(model[0]['ua'], context_t[0])
         )
        + model[0]['ba']
    )

    shortmem = F.tanh(
        model[0]['ws'] *
        (torch.matmul(model[0]['vs'], sequence_t) +
         attention * context_t[0]
         )
        + model[0]['bs']
    )

    state = remember * shortmem + (1-remember) * context_t[0]

    if drop_in != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * drop_in))

        for _ in drop: state[_] = Tensor(0)

    t_states.append(state)

    for _ in range(1, len(model)-1):

        # GRU Middle layer(s)

        try:

            remember = F.sigmoid(
                model[_]['wr'] *
                (torch.matmul(model[_]['vr'], t_states[-1]) +
                 model[_]['ur'] * context_t[_]
                 )
                + model[_]['br']
            )

            attention = F.sigmoid(
                model[_]['wa'] *
                (torch.matmul(model[_]['va'], t_states[-1]) +
                 model[_]['ua'] * context_t[_]
                 )
                + model[_]['ba']
            )

            shortmem = F.tanh(
                model[_]['ws'] *
                (torch.matmul(model[_]['vs'], t_states[-1]) +
                 attention * context_t[_]
                 )
                + model[_]['bs']
            )

        except:

            remember = F.sigmoid(
                model[_]['wr'] *
                (torch.matmul(model[_]['vr'], t_states[-1]) +
                 torch.matmul(model[_]['ur'] * context_t[_])
                 )
                + model[_]['br']
            )

            attention = F.sigmoid(
                model[_]['wa'] *
                (torch.matmul(model[_]['va'], t_states[-1]) +
                 torch.matmul(model[_]['ua'], context_t[_])
                 )
                + model[_]['ba']
            )

            shortmem = F.tanh(
                model[_]['ws'] *
                (torch.matmul(model[_]['vs'], t_states[-1]) +
                 attention * context_t[_]
                 )
                + model[_]['bs']
            )


        state = remember * shortmem + (1-remember) * context_t[_]

        if drop_mid != 0.0:
            drop = random.choices(range(len(state)), k=int(len(state) * drop_mid))
            for _ in drop: state[_] = 0

        t_states.append(state)

    # LSTM Output layer

    try:

        remember = F.sigmoid(
            model[-1]['wr'] *
            (torch.matmul(model[-1]['vr'], t_states[-1]) +
             model[-1]['ur'] * context_t[-1]
             )
            + model[-1]['br']
        )

        forget = F.sigmoid(
            model[-1]['wf'] *
            (torch.matmul(model[-1]['vf'], t_states[-1]) +
             model[-1]['uf'] * context_t[-1]
             )
            + model[-1]['bf']
        )

        attention = F.sigmoid(
            model[-1]['wa'] *
            (torch.matmul(model[-1]['va'], t_states[-1]) +
             model[-1]['ua'] * context_t[-1] +
             torch.matmul(model[-1]['ua2'], out_context_t[0]) +
             torch.matmul(model[-1]['wif'], t_states[0])
             )
            + model[-1]['ba']
        )

        shortmem = F.tanh(
            torch.matmul(model[-1]['ws'],
                         (torch.matmul(model[-1]['vs'], t_states[-1]) +
                          model[-1]['us'] * context_t[-1])
                         )
            + model[-1]['bs']
        )

    except:

        remember = F.sigmoid(
            model[-1]['wr'] *
            (torch.matmul(model[-1]['vr'], t_states[-1]) +
             model[-1]['ur'] * context_t[-1]
             )
            + model[-1]['br']
        )

        forget = F.sigmoid(
            model[-1]['wf'] *
            (torch.matmul(model[-1]['vf'], t_states[-1]) +
             model[-1]['uf'] * context_t[-1]
             )
            + model[-1]['bf']
        )

        attention = F.sigmoid(
            model[-1]['wa'] *
            (torch.matmul(model[-1]['va'], t_states[-1]) +
             model[-1]['ua'] * context_t[-1] +
             torch.matmul(model[-1]['ua2'], out_context_t) +
             torch.matmul(model[-1]['wif'], t_states[0])
             )
            + model[-1]['ba']
        )

        shortmem = F.tanh(
            torch.matmul(model[-1]['ws'],
                         (torch.matmul(model[-1]['vs'], t_states[-1]) +
                          model[-1]['us'] * context_t[-1])
                         )
            + model[-1]['bs']
        )


    state = remember * shortmem + forget * context_t[-1]
    t_states.append(state)

    outstate = attention * F.tanh(t_states[-1])

    if drop_out != 0.0:
        drop = random.choices(range(len(outstate)), k=int(len(outstate) * drop_out))

        for _ in drop: outstate[_] = Tensor(0)

    output = torch.sigmoid(torch.matmul(model[-1]['wo'], outstate) + model[-1]['bo'])

    return t_states, outstate, output


    #   Forward-Prop Method(s)


def forward_prop_train(model, sequence, context=None, gen_seed=None, gen_iterations=None, drop_in=0.0, drop_mid=0.0, drop_out=0.0):

    states = [context] if context is not None else Vanilla.init_states(model,hm_ins)
    out_states = Vanilla.init_outstates(model,hm_ins)
    outputs = []

    for t in range(len(sequence)):

        t_states, out_state, out = prop_func_alt(model, sequence[t], states[-1], out_states[-1], drop_in, drop_mid, drop_out)
        states.append(t_states)
        out_states.append(out_state)
        outputs.append(out)

    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]

    for t in range(gen_iterations):

        t_states, out_state, output = prop_func_alt(model, outputs[-1], states[-1], out_states[-1], drop_in, drop_mid, drop_out)
        states.append(t_states)
        out_states.append(out_state)
        outputs.append(output)

    del outputs[0]
    return outputs


def forward_prop_interact(model, sequence, context=None, gen_seed=None):

    states = [context] if context is not None else Vanilla.init_states(model,hm_ins)
    out_states = Vanilla.init_outstates(model,hm_ins)
    outputs = []

    for t in range(len(sequence)):

        t_states, out_state, out = Vanilla.prop_func(model, sequence[t], states[-1], out_states[-1])
        states.append(t_states)
        out_states.append(out_state)
        outputs.append(out)

    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]

    t = 0
    while not is_stop_cond(outputs[-1]) and t < MAX_PROP_TIME:

        t_states, out_state, output = Vanilla.prop_func(model, outputs[-1], states[-1], out_states[-1])
        write_neural_state(t_states)
        write_response(output)
        states.append(t_states)
        out_states.append(out_state)
        outputs.append(output)
        t += 1

    del outputs[0]
    return outputs



#     Model-Update_math     #



    # Applying Nesterov's Momentum #

# 1- create a model copy and update it w/ step1

def nesterov_step1(model_copy, moments, alpha=0.9):
    with torch.no_grad():
        for _,layer in enumerate(model_copy):
            for __,weight in enumerate(layer.values()):
                weight += alpha * moments[_][__]

# 2- forw_prop updated model copy and apply gradients to original model

# 3- update original model w/ step2 or step2_adaptive

def nesterov_step2(model, moments, batch_size=1, alpha=0.9, beta=0.1):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size

                    moments[_][__] = alpha * moments[_][__] + beta * weight.grad
                    weight += moments[_][__]
                    weight.grad = None

def nesterov_step2_adaptive(model, accugrads, moments, batch_size=1, alpha_moments=0.1, beta_moments=0.9, alpha_accugrads=0.1, lr=0.01):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size

                    moments[_][__] = alpha_moments * moments[_][__] + beta_moments * weight.grad
                    accugrads[_][__] = alpha_accugrads * accugrads[_][__] + (1 - alpha_accugrads) * weight.grad ** 2
                    weight += lr * moments[_][__] / (torch.sqrt(sum(accugrads[_][__])) + 1e-8)
                    weight.grad = None


def update_model_adam(model, accugrads, moments, epoch_nr, batch_size=1, lr=0.001, alpha_moments=0.9, alpha_accugrads=0.999):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size

                    moments[_][__] = alpha_moments * moments[_][__] + (1 - alpha_moments) * weight.grad
                    accugrads[_][__] = alpha_accugrads * accugrads[_][__] + (1 - alpha_accugrads) * weight.grad ** 2

                    moment_hat = moments[_][__] / (1 - alpha_moments ** epoch_nr)
                    accugrad_hat = accugrads[_][__] / (1 - alpha_accugrads ** epoch_nr)

                    weight += lr * moment_hat / (torch.sqrt(sum(accugrad_hat)) + 1e-8)
                    weight.grad = None



#     Helpers for Max-Time Accu_Grads     #


def init_accugrads_adv(model):
    accu_grads = []
    for layer in model:
        layer_grads = []
        for _ in layer.keys():
            layer_grads.append([None] * grad_save_time)
        accu_grads.append(layer_grads)
    return accu_grads

def save_accugrads_adv(accu_grads, model_id=None):
    model_id = '' if model_id is None else str(model_id)
    res.pickle_save(accu_grads,'model' + model_id + '_accugrads_time.pkl')

def load_accugrads_adv(model, model_id=None, from_basic_accugrads=False):  # = True for backward support
    model_id = '' if model_id is None else str(model_id)
    if not from_basic_accugrads:
        try:
            accu_grads_time = res.pickle_load('model' + model_id + '_accugrads_time.pkl')
            print('accugrads.pkl loaded.')
        except:
            print('accugrads: .pkl not found, initializing.')
            accu_grads_time = init_accugrads_adv(model) # , grad_save_time)
    else:
        accu_grads_time = init_accugrads_adv(model) # , grad_save_time)
        found_accugrads = glob.glob('model' + model_id + '_accugrads*.pkl')
        print(f'accugrads backward support: integrating {len(found_accugrads)} data(s)')
        for accu_grad in found_accugrads:
            accu_grad = res.pickle_load(accu_grad)
            for _,layer in enumerate(model):
                for __,weight in enumerate(layer.keys()):
                    accu_grads_time[_][__].pop(0)
                    accu_grads_time[_][__].append(accu_grad[_][__])
    return accu_grads_time


#     General Helpers     #


def write_neural_state(tstates):
    pass # todo: dont pass
    # with open('states.txt','a') as file:
    #     for _, layer in enumerate(tstates):
    #         states = []
    #         for state in layer:
    #             states.append([state{}])
    #
    #
    #             states.append(float(state.sum()))
    #         file.write(f'{_}:{states} \n')
    #         file.flush()
    #         os.fsync(file.fileno())

def write_response(response):

    respon = []
    for _ in range(res.vocab_size):
        respon.extend([response[0][_] +
                       response[1][_] +
                       response[2][_] +
                       response[3][_]])

    with open('response.txt','a') as file:
        for resp in respon:
            file.write(str(float(resp))+' ')
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())

def get_latest_response():
    try:
        with open('response.txt','a') as file:
            latest_response = file.readlines()[-1]
        return latest_response
    except: pass


#       Other


def plot_loss_txts(hm_mins_refresh=2):

    from matplotlib import style
    import matplotlib.pyplot as plot
    import matplotlib.animation as animation
    # import matplotlib.patches as mpatches
    import random

    loss_1_path = 'loss_1.txt'  # input('import loss_1: ')
    loss_2_path = 'loss_2.txt'  # input('import loss_2: ')
    loss_3_path = 'loss_3.txt'  # input('import loss_3: ')
    loss_4_path = 'loss_4.txt'  # input('import loss_4: ')

    fig = plot.figure()
    axis = fig.add_subplot(1, 1, 1)

    theme = random.choice(['Solarize_Light2', 'fivethirtyeight'])
    style.use(theme)

    def animate(i):
        epochs, losses = [], []
        
        with open(loss_1_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    epochs.append(int(epoch))
                    losses.append(int(loss))
                    
        with open(loss_2_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))

        with open(loss_3_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))
                    
        with open(loss_4_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))
                    
        axis.clear()
        axis.plot(epochs, losses[:len(epochs)], 'r', label='Vocabulary')
        axis.plot(epochs, losses[len(epochs) * 2: len(epochs) * 3], 'b', label='Rhythm')
        axis.plot(epochs, losses[len(epochs):len(epochs) * 2], 'g', label='Octaves')
        axis.plot(epochs, losses[len(epochs) * 3: len(epochs) * 4], 'y', label='Velocities')

        plot.legend(bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=0.)

    ani = animation.FuncAnimation(fig, animate, hm_mins_refresh)
    var = plot.gcf()
    var.canvas.set_window_title('Loss Plot')

    plot.show()


# import multiprocessing
# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.Pool):
#     Process = NoDaemonProcess

if __name__ == '__main__':
    plot_loss_txts()
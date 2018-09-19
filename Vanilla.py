import torch

hm_ins = 4 ; hm_outs = 4
# max_time = 40

def create_model(in_size, layer_sizes, out_size):
    model = []

    #   GRU Input Layer

    model.append(
        {'vr':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'ur_1':torch.randn([layer_sizes[0]], requires_grad=True),
         'ur_2':torch.randn([layer_sizes[0]], requires_grad=True),
         'ur_3':torch.randn([layer_sizes[0]], requires_grad=True),
         'ur_4':torch.randn([layer_sizes[0]], requires_grad=True),
         'wr':torch.randn([layer_sizes[0]], requires_grad=True),
         'br':torch.zeros([layer_sizes[0]], requires_grad=True),

         'va':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'ua_1':torch.randn([layer_sizes[0]], requires_grad=True),
         'ua_2':torch.randn([layer_sizes[0]], requires_grad=True),
         'ua_3':torch.randn([layer_sizes[0]], requires_grad=True),
         'ua_4':torch.randn([layer_sizes[0]], requires_grad=True),
         'wa':torch.randn([layer_sizes[0]], requires_grad=True),
         'ba':torch.zeros([layer_sizes[0]], requires_grad=True),

         'vs_1':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'vs_2':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'vs_3':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'vs_4':torch.randn([layer_sizes[0], in_size], requires_grad=True),
         'ws_1':torch.randn([layer_sizes[0]], requires_grad=True),
         'ws_2':torch.randn([layer_sizes[0]], requires_grad=True),
         'ws_3':torch.randn([layer_sizes[0]], requires_grad=True),
         'ws_4':torch.randn([layer_sizes[0]], requires_grad=True),
         'bs':torch.zeros([layer_sizes[0]], requires_grad=True)

         })

    #   GRU Middle layer(s)

    for _ in range(1, len(layer_sizes) - 1):

        prev_lsize = layer_sizes[_-1]

        model.append(
            {'vr_1':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vr_2':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vr_3':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vr_4':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'ur':torch.randn([layer_sizes[_]], requires_grad=True),
             'wr':torch.randn([layer_sizes[_]], requires_grad=True),
             'br':torch.zeros([layer_sizes[_]], requires_grad=True),

             'va_1':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'va_2':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'va_3':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'va_4':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'ua':torch.randn([layer_sizes[_], layer_sizes[_]], requires_grad=True),
             'wa':torch.randn([layer_sizes[_]], requires_grad=True),
             'ba':torch.zeros([layer_sizes[_]], requires_grad=True),

             'vs_1':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vs_2':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vs_3':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'vs_4':torch.randn([layer_sizes[_], prev_lsize], requires_grad=True),
             'ws':torch.randn([layer_sizes[_]], requires_grad=True),
             'bs':torch.zeros([layer_sizes[_]], requires_grad=True)

             })

    #   LSTM Output layer

    model.append(
        {'vr_1':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vr_2':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vr_3':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vr_4':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'ur':torch.randn([layer_sizes[-1]], requires_grad=True),
         'wr':torch.randn([layer_sizes[-1]], requires_grad=True),
         'br':torch.zeros([layer_sizes[-1]], requires_grad=True),

         'va_1':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'va_2':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'va_3':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'va_4':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'ua':torch.randn([layer_sizes[-1]], requires_grad=True),
         'ua2_1':torch.randn([layer_sizes[-1], layer_sizes[-1]], requires_grad=True),
         'ua2_2':torch.randn([layer_sizes[-1], layer_sizes[-1]], requires_grad=True),
         'ua2_3':torch.randn([layer_sizes[-1], layer_sizes[-1]], requires_grad=True),
         'ua2_4':torch.randn([layer_sizes[-1], layer_sizes[-1]], requires_grad=True),
         'wa':torch.randn([layer_sizes[-1]], requires_grad=True),
         'ba':torch.zeros([layer_sizes[-1]], requires_grad=True),

         'vs_1':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vs_2':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vs_3':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vs_4':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'us':torch.randn([layer_sizes[-1]], requires_grad=True),
         'ws':torch.randn([layer_sizes[-1], layer_sizes[-1]], requires_grad=True),
         'bs':torch.zeros([layer_sizes[-1]], requires_grad=True),

         'vf_1':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vf_2':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vf_3':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'vf_4':torch.randn([layer_sizes[-1], layer_sizes[-2]], requires_grad=True),
         'uf':torch.randn([layer_sizes[-1]], requires_grad=True),
         'wf':torch.randn([layer_sizes[-1]], requires_grad=True),
         'bf':torch.zeros([layer_sizes[-1]], requires_grad=True),

         'wif_1':torch.randn([layer_sizes[-1], layer_sizes[0]], requires_grad=True),
         'wif_2':torch.randn([layer_sizes[-1], layer_sizes[0]], requires_grad=True),
         'wif_3':torch.randn([layer_sizes[-1], layer_sizes[0]], requires_grad=True),
         'wif_4':torch.randn([layer_sizes[-1], layer_sizes[0]], requires_grad=True),

         'wo_1':torch.randn([out_size, layer_sizes[-1]], requires_grad=True),
         'wo_2':torch.randn([out_size, layer_sizes[-1]], requires_grad=True),
         'wo_3':torch.randn([out_size, layer_sizes[-1]], requires_grad=True),
         'wo_4':torch.randn([out_size, layer_sizes[-1]], requires_grad=True),

         'bo_1':torch.zeros([out_size], requires_grad=True),
         'bo_2':torch.zeros([out_size], requires_grad=True),
         'bo_3':torch.zeros([out_size], requires_grad=True),
         'bo_4':torch.zeros([out_size], requires_grad=True)

         })

    return model


    # Forward-Prop_math


def prop_func(model, sequence_t, context_t, out_context_t):
    t_states = []

    # GRU Input Layer

    remember = torch.sigmoid(
        model[0]['wr'] *
        (torch.matmul(model[0]['vr'], sequence_t[0]) +
         torch.matmul(model[0]['vr'], sequence_t[1]) +
         torch.matmul(model[0]['vr'], sequence_t[2]) +
         torch.matmul(model[0]['vr'], sequence_t[3]) +

         model[0]['ur_1'] * context_t[0][0] +
         model[0]['ur_2'] * context_t[0][1] +
         model[0]['ur_3'] * context_t[0][2] +
         model[0]['ur_4'] * context_t[0][3]
         )
        + model[0]['br']
    )

    attention = torch.sigmoid(
        model[0]['wa'] *
        (torch.matmul(model[0]['va'], sequence_t[0]) +
         torch.matmul(model[0]['va'], sequence_t[1]) +
         torch.matmul(model[0]['va'], sequence_t[2]) +
         torch.matmul(model[0]['va'], sequence_t[3]) +

         model[0]['ua_1'] * context_t[0][0] +
         model[0]['ua_2'] * context_t[0][1] +
         model[0]['ua_3'] * context_t[0][2] +
         model[0]['ua_4'] * context_t[0][3]
         )
        + model[0]['ba']
    )

    shortmem_1 = torch.tanh(
        model[0]['ws_1'] *
        (torch.matmul(model[0]['vs_1'], sequence_t[0]) +
         attention * context_t[0][0]
         )
        + model[0]['bs']
    )

    shortmem_2 = torch.tanh(
        model[0]['ws_2'] *
        (torch.matmul(model[0]['vs_2'], sequence_t[0]) +
         attention * context_t[0][1]
         )
        + model[0]['bs']
    )

    shortmem_3 = torch.tanh(
        model[0]['ws_3'] *
        (torch.matmul(model[0]['vs_3'], sequence_t[0]) +
         attention * context_t[0][2]
         )
        + model[0]['bs']
    )

    shortmem_4 = torch.tanh(
        model[0]['ws_4'] *
        (torch.matmul(model[0]['vs_4'], sequence_t[0]) +
         attention * context_t[0][3]
         )
        + model[0]['bs']
    )

    state_1 = remember * shortmem_1 + (1-remember) * context_t[0][0]
    state_2 = remember * shortmem_2 + (1-remember) * context_t[0][1]
    state_3 = remember * shortmem_3 + (1-remember) * context_t[0][2]
    state_4 = remember * shortmem_4 + (1-remember) * context_t[0][3]

    t_states.append([state_1,state_2,state_3,state_4])

    # GRU Middle layer

    remember = torch.sigmoid(
        model[1]['wr'] *
        (torch.matmul(model[1]['vr_1'], t_states[-1][0]) +
         torch.matmul(model[1]['vr_2'], t_states[-1][1]) +
         torch.matmul(model[1]['vr_3'], t_states[-1][2]) +
         torch.matmul(model[1]['vr_4'], t_states[-1][3]) +
         model[1]['ur'] * context_t[1]
         )
        + model[1]['br']
    )

    attention = torch.sigmoid(
        model[1]['wa'] *
        (torch.matmul(model[1]['va_1'], t_states[-1][0]) +
         torch.matmul(model[1]['va_2'], t_states[-1][1]) +
         torch.matmul(model[1]['va_3'], t_states[-1][2]) +
         torch.matmul(model[1]['va_4'], t_states[-1][3]) +
         torch.matmul(model[1]['ua'], context_t[1])
         )
        + model[1]['ba']
    )

    shortmem = torch.tanh(
        model[1]['ws'] *
        (torch.matmul(model[1]['vs_1'], t_states[-1][0]) +
         torch.matmul(model[1]['vs_2'], t_states[-1][1]) +
         torch.matmul(model[1]['vs_3'], t_states[-1][2]) +
         torch.matmul(model[1]['vs_4'], t_states[-1][3]) +
         attention * context_t[1]
         )
        + model[1]['bs']
    )

    state = remember * shortmem + (1-remember) * context_t[1]
    t_states.append(state)

    # LSTM Output layer

    remember = torch.sigmoid(
        model[-1]['wr'] *
        (torch.matmul(model[-1]['vr_1'], t_states[-1]) +
         torch.matmul(model[-1]['vr_2'], t_states[-1]) +
         torch.matmul(model[-1]['vr_3'], t_states[-1]) +
         torch.matmul(model[-1]['vr_4'], t_states[-1]) +
         model[-1]['ur'] * context_t[-1]
         )
        + model[-1]['br']
    )

    forget = torch.sigmoid(
        model[-1]['wf'] *
        (torch.matmul(model[-1]['vf_1'], t_states[-1]) +
         torch.matmul(model[-1]['vf_2'], t_states[-1]) +
         torch.matmul(model[-1]['vf_3'], t_states[-1]) +
         torch.matmul(model[-1]['vf_4'], t_states[-1]) +
         model[-1]['uf'] * context_t[-1]
         )
        + model[-1]['bf']
    )

    attention_1 = torch.sigmoid(
        model[-1]['wa'] *
        (torch.matmul(model[-1]['va_1'], t_states[-1]) +
         model[-1]['ua'] * context_t[-1] +
         torch.matmul(model[-1]['ua2_1'], out_context_t[0]) +
         torch.matmul(model[-1]['wif_1'], t_states[0][0])
         )
        + model[-1]['ba']
    )

    attention_2 = torch.sigmoid(
        model[-1]['wa'] *
        (torch.matmul(model[-1]['va_2'], t_states[-1]) +
         model[-1]['ua'] * context_t[-1] +
         torch.matmul(model[-1]['ua2_2'], out_context_t[1]) +
         torch.matmul(model[-1]['wif_2'], t_states[0][1])
         )
        + model[-1]['ba']
    )

    attention_3 = torch.sigmoid(
        model[-1]['wa'] *
        (torch.matmul(model[-1]['va_3'], t_states[-1]) +
         model[-1]['ua'] * context_t[-1] +
         torch.matmul(model[-1]['ua2_3'], out_context_t[2]) +
         torch.matmul(model[-1]['wif_3'], t_states[0][2])
         )
        + model[-1]['ba']
    )

    attention_4 = torch.sigmoid(
        model[-1]['wa'] *
        (torch.matmul(model[-1]['va_4'], t_states[-1]) +
         model[-1]['ua'] * context_t[-1] +
         torch.matmul(model[-1]['ua2_4'], out_context_t[3]) +
         torch.matmul(model[-1]['wif_4'], t_states[0][3])
         )
        + model[-1]['ba']
    )

    shortmem = torch.tanh(
        torch.matmul(model[-1]['ws'],
                     (torch.matmul(model[-1]['vs_1'], t_states[-1]) +
                      torch.matmul(model[-1]['vs_2'], t_states[-1]) +
                      torch.matmul(model[-1]['vs_3'], t_states[-1]) +
                      torch.matmul(model[-1]['vs_4'], t_states[-1]) +
                      model[-1]['us'] * context_t[-1])
                     )
        + model[-1]['bs']
    )


    state = remember * shortmem + forget * context_t[-1]
    t_states.append(state)

    outstate_1 = attention_1 * torch.tanh(t_states[-1])
    outstate_2 = attention_2 * torch.tanh(t_states[-1])
    outstate_3 = attention_3 * torch.tanh(t_states[-1])
    outstate_4 = attention_4 * torch.tanh(t_states[-1])

    output1 = torch.sigmoid(torch.matmul(model[-1]['wo_1'], outstate_1) + model[-1]['bo_1'])
    output2 = torch.matmul(model[-1]['wo_2'], outstate_2) + model[-1]['bo_2']
    output3 = torch.matmul(model[-1]['wo_3'], outstate_3) + model[-1]['bo_3']
    output4 = torch.matmul(model[-1]['wo_4'], outstate_4) + model[-1]['bo_4']

    return t_states, [outstate_1, outstate_2, outstate_3, outstate_4], [output1, output2, output3, output4]


#   Forward-Prop_method


def forward_prop(model, sequence, context=None, gen_seed=None, gen_iterations=None):


    #   listen


    vocab_seq, oct_seq, dur_seq, vol_seq  = sequence
    states = [context] if context is not None else init_states(model, hm_ins=hm_ins)
    out_states = init_outstates(model, hm_outs=hm_outs)
    outputs = []

    for t in range(len(sequence[0])):

        t_states, out_state, out = prop_func(model, [vocab_seq[t], oct_seq[t], dur_seq[t], vol_seq[t]], states[-1], out_states[-1])
        states.append(t_states)
        out_states.append(out_state)
        outputs.append(out)


    #   generate


    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]     # sequence, an array of outputs, where out[0] = vocab, out[1] = rhythm

    if gen_iterations is None:
        pass
        # t = 0
        # while t < max_time:
        #     print('------',outputs[-1])
        #     t_states, out_state, output = prop_func(model, outputs[-1], states[-1], out_states[-1])
        #     states.append(t_states)
        #     out_states.append(out_state)
        #     outputs.append(output)
        #     t += 1

    else:

        for t in range(gen_iterations):

            t_states, out_state, output = prop_func(model, outputs[-1], states[-1], out_states[-1])
            states.append(t_states)
            out_states.append(out_state)
            outputs.append(output)


    del outputs[0]
    return outputs


#   helpers


def custom_softmax(output_seq):
    return (lambda e_x: e_x / e_x.sum())(torch.exp(output_seq))

def custom_entropy(output_seq, label_seq, will_softmax=True):
    sequence_losses = []

    for t in range(len(label_seq)):
        lbl = custom_softmax(label_seq[t]) if will_softmax else label_seq[t]
        pred = custom_softmax(output_seq[t]) if will_softmax else output_seq[t]

        sequence_losses.append((lbl * torch.log(pred)).mean())

    return sequence_losses

def custom_mse(output_seq, label_seq):
    sequence_losses = []

    for t in range(len(label_seq)):
        lbl = label_seq[t]
        pred = output_seq[t]

        sequence_losses.append((lbl - pred).mean()) # why not ; lbl - pred ** 3     -faster convergence?

    return sequence_losses




def init_states(model, hm_ins):
    states_t0 = [[torch.ones([len(model[0]['vr'])], requires_grad=True) for _ in range(hm_ins)]]
    for _ in range(1,len(model)-1):
        states_t0.append(torch.ones([len(model[_]['vr_1'])], requires_grad=True))
    states_t0.append(torch.ones([len(model[-1]['vr_1'])], requires_grad=True))
    return [states_t0]

def init_outstates(model, hm_outs):
    return [[torch.ones([len(model[-1]['vr_1'])], requires_grad=True) for _ in range(hm_outs)]]

def stop_cond(output_t):

    notes = output_t[0]
    durations = output_t[2]

    sel_notes = [_ for _,e in enumerate(notes) if e.item() >= 0.5]
    if sel_notes == []: sel_notes = [torch.argmax(notes).item()]

    for note in sel_notes:
        if note == 12: return True

    for dur in durations:
        if float(dur) >= 2: return True
    return False



        # External Helpers / Optimizers



def update_gradients(loss_nodes):
    for node in loss_nodes:
        for timestep in node:
            timestep.backward(retain_graph=True)


def update_model(model, batch_size=1, learning_rate=0.001):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight += learning_rate * weight.grad / batch_size
                    weight.grad = None


def update_model_momentum(model, moments, batch_size=1, alpha=0.9, beta=0.1):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    moments[_][__] = alpha * moments[_][__] + beta * weight.grad
                    weight += moments[_][__]
                    weight.grad = None


def update_model_rmsprop(model, accu_grads, batch_size=1, lr=0.01, alpha=0.9):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    accu_grads[_][__] = alpha * accu_grads[_][__] + (1 - alpha) * weight.grad**2
                    weight += lr * weight.grad / (torch.sqrt(accu_grads[_][__]) + 1e-8)
                    weight.grad = None



    # Additional Helpers


def disp_params(model):
    for _,layer in enumerate(model):
        for weight in layer:
            print('Layer',_,'Weight',weight,layer[weight])

def disp_grads(model):
    for _,layer in enumerate(model):
            for name in layer:
                print('Layer',_,'Grad',name,layer[name].grad)


        #   Multiprocess Helpers


def return_weights(model):
    params = []
    names = []
    for _,layer in enumerate(model):
        for w in layer:
            params.append(layer[w])
            names.append(f'Layer {_} {w} :')
    return names, params

def return_grads(model):
    grads = []
    for _,layer in enumerate(model):
        for w in layer:
            grads.append(layer[w].grad)
    return grads

def apply_grads(model, grads):
    ctr = 0
    for _,layer in enumerate(model):
        for w in layer:
            this_grad = grads[ctr]
            if layer[w].grad is None: layer[w].grad = this_grad
            else: layer[w].grad += this_grad

            ctr +=1


    # MODEL CONVERSION


class TorchModel(torch.nn.Module):
    def __init__(self, model):
        super(TorchModel, self).__init__()
        self.model = model

    def forward(self, inp):
        return forward_prop(self.model, inp)

def model2torch(model):
    tm = TorchModel(model)
    return tm

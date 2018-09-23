# import Vanilla

import res
import utils
from utils                      \
    import forward_prop_interact \
    as ask                        \

# import time
import torch
from torch                            \
    import Tensor

# from multiprocessing                    \
#     import Process, Manager

max_octave = res.MAX_OCTAVE
max_duration = res.MAX_DURATION
max_volume = res.MAX_VOLUME





def bootstrap():


    model = res.load_model()
    while model is None:
        model_id = input('Import model and Hit Enter.. ')
        if model_id == '':
            model = res.load_model()
        else:
            model = res.load_model(model_id)


    chord_mode = input("hit 's' to Quit Chord Mode: ")
    chord_mode = False if chord_mode == 's' else True


    while True:

        open('states.txt', "w+").close()
        open('response.txt', "w+").close()

        inp_len = int(input('enter an Input Length: '))
        input_sequence = get_user_input(inp_len)


        print('Processing..')

        #     # Process 1
        #
        # proc1 = Process(target=ask, args=(model, input_sequence))
        # proc1.start()
        #
        #     # Process 2
        #
        # proc2 = Process(target=print_response, args=(chord_mode,))
        # proc2.start()
        #
        #     # handle
        #
        # proc1.join()
        # proc2.terminate()
        #
        #     # end of program
        #
        # for result in manager.dict().keys():
        #     print(result)
        #
        # input('Response processed.')

        responses = ask(model, input_sequence)

        converted_response = [ai_2_human(out, chordMode=chord_mode) for out in responses]

        for response in converted_response:

            print('---')
            print(' Notes:', [res.note_reverse_dict[_] for _ in response[0]])
            print(' Octaves:', response[1])
            print(' Durations:', response[2])
            print(' Velocities:', response[3])
            print('---')

        print(f'Response length: {len(converted_response)}')

        input()





# converters


def ai_2_human(out_t, chordMode=True):

    len_out = len(out_t) # todo replace with Vanilla.IOsize = 52
    vocabs, octaves, durations, volumes = out_t[:int(len_out/4)], \
                                          out_t[int(len_out/4):int(len_out/2)], \
                                          out_t[int(len_out/2):int(len_out/4*3)], \
                                          out_t[int(len_out/4*3):]

    # sel_vocabs = []
    sel_octs   = []
    sel_durs   = []
    sel_vols   = []

    if chordMode:
        sel_vocabs = [_ for _,e in enumerate(vocabs) if e.item() >= 0.1]
    else:
        sel_vocabs = [torch.argmax(vocabs).item()]

    for vocab in sel_vocabs:
        sel_octs.append(round(float(octaves[vocab]) * max_octave))
        sel_durs.append(round(float(durations[vocab]) * max_duration))
        sel_vols.append(round(float(volumes[vocab]) * max_volume))

    return sel_vocabs, sel_octs, sel_durs, sel_vols


def human_2_ai(data):

    notes, octaves, durations, volumes = data
    inp_len = len(notes)

    c_notes = [res.note_dict[notes[i].upper()] for i in range(inp_len)]

    vocab_vect = res.empty_vect.copy()
    oct_vect   = res.empty_vect.copy()
    dur_vect   = res.empty_vect.copy()
    vol_vect   = res.empty_vect.copy()

    for i, note in enumerate(c_notes):
        duplicate_note = False if vocab_vect[note] == 0 else True
        vocab_vect[note] += 1
        oct_vect[note]   += float(octaves[i])
        dur_vect[note]   += float(durations[i])
        vol_vect[note]   += float(volumes[i])
        if duplicate_note:
            oct_vect[note]   /= 2
            dur_vect[note]   /= 2
            vol_vect[note]   /= 2

    # normalization & fixes

    vocab_sum = sum(vocab_vect)

    if vocab_sum != 1: vocab_vect = [e/vocab_sum for e in vocab_vect]
    oct_vect = [e/max_octave for e in oct_vect]
    dur_vect = [e/max_duration for e in dur_vect]
    vol_vect = [e/max_volume for e in vol_vect]

    return vocab_vect, oct_vect, dur_vect, vol_vect


# other helpers


def get_user_input(inp_len):
    vocab_seq = []
    oct_seq   = []
    dur_seq   = []
    vol_seq   = []

    for i in range(inp_len):
        notes = str(input('Enter a tone / chord : ')).split(' ')
        octs = str(input('Enter octaves : ')).split(' ')
        durs = str(input('Enter durations : ')).split(' ')
        vols = str(input('Enter volumes : ')).split(' ')

        data = [notes, octs, durs, vols]
        vocab_vect, oct_vect, dur_vect, vol_vect = human_2_ai(data)

        vocab_seq.append(vocab_vect)
        oct_seq.append(oct_vect)
        dur_seq.append(dur_vect)
        vol_seq.append(vol_vect)

    sequence = []

    for _ in range(len(vocab_seq)):
        vec_t = []
        [vec_t.extend(e) for e in [vocab_seq[_], oct_seq[_], dur_seq[_], vol_seq[_]]]

        sequence.append(Tensor(vec_t))


    return sequence


def print_response(chord_mode):
    latest_response = None

    while True:

        response = utils.get_latest_response()

        if response != latest_response:

            converted_response = ai_2_human(response, chordMode=chord_mode)

            print('---')
            print(' Notes:',converted_response[0])
            print(' Octaves:',converted_response[1])
            print(' Durations:',converted_response[2])
            print(' Velocities:',converted_response[3])
            print('---')

            latest_response = response

        # time.sleep(0.2)





if __name__ == '__main__':
    bootstrap()


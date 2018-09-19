import glob
import pickle
import random
import music21

from music21 import converter # note, chord
from multiprocessing import Pool, cpu_count

min_seq_len = 10
max_seq_len = 40

MAX_OCTAVE = 7
MAX_DURATION = 8.0 ; SPLIT_DURATION = 2.0
MAX_VOLUME = 127


show_passed_exceptions = True


def preprocess():

    vocab_seqs_X, vocab_seqs_Y = [], []
    oct_seqs_X,   oct_seqs_Y   = [], []
    dur_seqs_X,   dur_seqs_Y   = [], []
    vol_seqs_X,   vol_seqs_Y   = [], []

    raw_files = glob.glob("samples/*.mid")
    imported_files = []

    global show_passed_exceptions
    if show_passed_exceptions and \
       input('Show Passed Errors: (Y/N)').lower() != 'y':
        show_passed_exceptions = False

    print(f'\nDetected CPU(s): {cpu_count()}\n')

    with Pool(cpu_count()) as p:

        results = p.map_async(import_fn, raw_files)

        p.close()
        p.join()

        for result in results.get():
            if result is not None:
                imported_files.append(result)
        print(f'Files Obtained: {len(results.get())}\n')


    with Pool(cpu_count()) as p2:

        results = p2.map_async(parse_fn, imported_files)

        p2.close()
        p2.join()

        for result in results.get():
            if len(result[0]) is not 0:
                vocab_seqs_X.extend(result[0]) ; vocab_seqs_Y.extend(result[1])
                oct_seqs_X.extend(result[2])   ; oct_seqs_Y.extend(result[3])
                dur_seqs_X.extend(result[4])   ; dur_seqs_Y.extend(result[5])
                vol_seqs_X.extend(result[6])   ; vol_seqs_Y.extend(result[7])
        print()

    print(f'Samples Collected: {len(vocab_seqs_X)}\n')

    data = [
        [vocab_seqs_X, oct_seqs_X, dur_seqs_X, vol_seqs_X],
        [vocab_seqs_Y, oct_seqs_Y, dur_seqs_Y, vol_seqs_Y]]

    return data


def parse_fn(stream):

    vocab_seqs_X, vocab_seqs_Y = [], []
    oct_seqs_X,   oct_seqs_Y   = [], []
    dur_seqs_X,   dur_seqs_Y   = [], []
    vol_seqs_X,   vol_seqs_Y   = [], []

    mstream = []

    vocab_seq_container = []
    oct_seq_container   = []
    dur_seq_container   = []
    vol_seq_container   = []

    for element in stream:

        vocab_vect, oct_vect, dur_vect, vol_vect = vectorize_element(element)

        if vocab_vect is not None:

            vocab_seq_container.append(vocab_vect)
            oct_seq_container.append(oct_vect)
            dur_seq_container.append(dur_vect)
            vol_seq_container.append(vol_vect)

            if split_cond(dur_vect):
                if min_seq_len <= len(vocab_seq_container) <= max_seq_len:
                    mstream.append([vocab_seq_container,
                                    oct_seq_container,
                                    dur_seq_container,
                                    vol_seq_container])

                vocab_seq_container, oct_seq_container, dur_seq_container, vol_seq_container = [], [], [], []


    for i, thing in enumerate(mstream[:-1]):
        thingp1 = mstream[i+1]
        vocab_seqs_X.append(thing[0])  ;oct_seqs_X.append(thing[1])  ;dur_seqs_X.append(thing[2])  ;vol_seqs_X.append(thing[3])
        vocab_seqs_Y.append(thingp1[0]);oct_seqs_Y.append(thingp1[1]);dur_seqs_Y.append(thingp1[2]);vol_seqs_Y.append(thingp1[3])


    # print('File Parsed.')
    return                         \
        vocab_seqs_X, vocab_seqs_Y, \
        oct_seqs_X,   oct_seqs_Y,    \
        dur_seqs_X,   dur_seqs_Y,     \
        vol_seqs_X,   vol_seqs_Y


def vectorize_element(element):

    vocab_vect = [0 for _ in range(vocab_size)]
    oct_vect   = vocab_vect.copy()
    dur_vect   = vocab_vect.copy()
    vol_vect   = vocab_vect.copy()

    try:
        if element.isNote:
            note_id = note_dict[element.pitch.name]
            if duration_isValid(element):
                vocab_vect[note_id] += 1
                oct_vect[note_id] += float(element.pitch.octave)
                dur_vect[note_id] += float(element.duration.quarterLength)
                vol_vect[note_id] += float(element.volume.velocity)

        elif element.isChord:
            for e in element:
                note_id = note_dict[e.pitch.name]
                if duration_isValid(e):
                    duplicateNote = vocab_vect[note_id] != 0
                    vocab_vect[note_id] += 1
                    oct_vect[note_id] += float(e.pitch.octave)
                    dur_vect[note_id] += float(e.duration.quarterLength)
                    vol_vect[note_id] += float(e.volume.velocity)

                    if duplicateNote:
                        oct_vect[note_id] /=2
                        dur_vect[note_id] /=2
                        vol_vect[note_id] /=2

        elif element.isRest:
            if duration_isValid(element):
                note_id = note_dict['R']
                vocab_vect[note_id] += 1
                dur_vect[note_id] += float(element.duration.quarterLength)


        # normalization & fixes

        vocab_sum = sum(vocab_vect)

        if vocab_sum == 0: return None, None, None, None

        if vocab_sum != 1: vocab_vect = [float(e/vocab_sum) for e in vocab_vect]
        # oct_vect = [float(e/MAX_OCTAVE) for e in oct_vect if e != 0]
        # dur_vect = [float(e/MAX_DURATION) for e in dur_vect if e != 0]
        # vol_vect = [float(e/MAX_VOLUME) for e in vol_vect if e != 0]

    except Exception as e:
        if show_passed_exceptions: print('Element', element, 'passed Error:', e)
        return None, None, None, None

    return vocab_vect, oct_vect, dur_vect, vol_vect


def duration_isValid(element): return 0.0 < float(element.duration.quarterLength) <= MAX_DURATION


note_dict = {
    'A' : 0,
    'A#': 1, 'B-': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4, 'D-': 4,
    'D' : 5,
    'D#': 6, 'E-': 6,
    'E' : 7,
    'F' : 8,
    'F#': 9, 'G-': 9,
    'G' :10,
    'G#':11, 'A-': 11,
    'R' :12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10:'G',
    11:'G#',
    12:'R'
}

vocab_size = len(note_reverse_dict)


def split_cond(dur_vect):
    for dur in dur_vect:
        if dur >= SPLIT_DURATION: return True
    return False

def import_fn(raw_file):
        try:
            raw_stream = converter.parse(raw_file)
            stream = ready_stream(raw_stream)
            return stream
        except:
            return None
        # finally: print('file scanned.')

def ready_stream(stream):
    # def_mtr = music21.meter.TimeSignature('4/4')
    # for element in stream: # todo: find a way to auto-conv everything to 4/4
    #     if type(element) is music21.meter.TimeSignature:
    #         del element
    # stream.insert(0, def_mtr)
    return stream.flat.elements


empty_vect = [0 for _ in range(vocab_size)]



    #   rest is dev purposes



class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size



    # Global Helpers #



def pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))



def write_epoch_details(epoch, epoch_losses):
    with open('summ_epochs.txt','a') as file:
        file.write(f"Epoch: {epoch}, Loss: {epoch_losses}\n")

    json_print_loss(epoch_losses)

def write_grad_details(model, floydout=False):
    try:
        grad_file = '/output/summ_grads.txt' if floydout else 'summ_grads.txt'
        if floydout:
            with open(grad_file,'a') as file:
                for _,layer in enumerate(model):
                    for name in layer:
                        file.write('Layer '+str(_)+' Grad '+name+' : '+str(layer[name].grad)+'\n')
                file.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
    except:
        print('Gradient save error.')

def json_print_loss(epoch_losses):
    for i, loss in enumerate(epoch_losses):
        print('{{"metric": "Loss {}", "value": {}}}'.format(i+1, float(loss)))
    print("")



def save_model(model, model_id=None, asText=False):
    model_id = '' if model_id is None else str(model_id)
    try:
        if not asText: pickle_save(model,'model' + model_id + '.pkl')
        else:
            with open('summ_models.txt','a') as file:
                file.write(f"> Epoch : {model_id} Parameters \n")
                for i, layer in enumerate(model):
                    file.write(f"Layer : {i} \n")
                    for key in layer:
                        file.write(key+" "+str(layer[key])+'\n')
                file.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
    except: print('Model save error.')

def load_model(model_id=None):
    model_id = '' if model_id is None else str(model_id)
    model = None
    try:
        model = pickle_load('model' + model_id + '.pkl')
        print('> Model loaded.')
    except:
        print('> Model not found.')
    finally: return model



def load_data(data_path, limit_size):

    dataset = pickle_load(data_path)

    sample_X, sample_Y = dataset
    vocab_X, oct_X, dur_X, vol_X = sample_X
    vocab_Y, oct_Y, dur_Y, vol_Y = sample_Y

    blocks = []
    for _ in range(len(vocab_X)):
        blocks.append([vocab_X[_], oct_X[_], dur_X[_], vol_X[_],
                       vocab_Y[_], oct_Y[_], dur_Y[_], vol_Y[_]])

    data = random.choices(blocks, k=limit_size)
    return data



#   API for ai -> midi



def smth(): pass


if __name__ == '__main__':
    data = preprocess()
    pickle_save(data,'samples.pkl')



# music21 basic guide:
#   for element in stream:
# element.pitch.name
# element.pitch.octave
# element.duration.quarterLength
# element.volume.velocity

import Vanilla

import trainer
import res
# import utils

import torch
import numpy as np


# import multiprocessing            # multi-model training
# from torch.multiprocessing   \
#     import Pool
# from concurrent.futures        \
#     import ThreadPoolExecutor
# from multiprocessing.pool import ThreadPool
# trainer.mPool = ThreadPool


    # parent details

hm_models = 1

max_success = 5
max_loss_inc_delta = 10
max_loss_inc_below_delta = 5
max_loss_inc_above_delta = 2
max_branch_n_data_reset = 10

model_architectures = [[2, 1, 3]]


    # data details

data_path = 'samples.pkl'
data_size = 2_000
batch_size = 100


    # training details

learning_rate_1 = 0.001
learning_rate_2 = 0.01

trainer.drop_in  = 0.0
trainer.drop_mid = 0.0
trainer.drop_out = 0.0

start_advanced = False

permanent_metadata = data_size >= 10_000

reducing_batch_sizes = True
reduce_batch_per_save = 5

# iterating_trainers = True
# iterate_trainer_per_save = 20


    # global conditions

real_losses = [[] for _ in range(hm_models)]

loss_initial = \
    np.array([999_999,999_999,999_999]) # ,999_999])

train_fns = [trainer.train_rms,
             trainer.train_rmsadv,
             trainer.train_adam]

primary_trainer = 0 ; secondary_trainer = 1



    #



def start_parenting(model_id=None):

    # initial condition(s) - per model

    ctr_successful_datasets = 0
    ctr_save_id = 0

    # ctr_epoch = 1
    ctr_success = 0
    ctr_loss_increase_below_delta = 0
    ctr_loss_increase_above_delta = 0
    ctr_branch_n_data_reset = 0

    checkpoints = []
    branches = []

    data = res.load_data(data_path, data_size)
    train_fn = train_fns[primary_trainer]
    trainer.learning_rate = learning_rate_1


    # load / create model files

    if model_id is None: model_id = 0
    model, accu_grads, _ = load_model_data(model_id)

    # set the first checkpoints

    checkpoints.append([model, accu_grads, loss_initial])
    branches.append([model, accu_grads, loss_initial])

    # start parenting

    while True:

        model = branches[-1][0] ; accu_grads = branches[-1][1] ; prev_loss = branches[-1][-1]

        model, accu_grads, loss_epochs = train_fn(model, accu_grads, data)

        current_loss = np.array(loss_epochs[0][0:-1])

        if all(current_loss < prev_loss):

            print(f'\n >> Epoch successful, Checkpoint created : {ctr_success+1}/{max_success} \n')

            ctr_success +=1
            ctr_loss_increase_below_delta = 0
            ctr_loss_increase_above_delta = 0

            res.write_loss(loss_epochs[0])
            real_losses[model_id].append(current_loss)

            checkpoints.append([model, accu_grads, current_loss])
            branches.append([model, accu_grads, current_loss])

            if ctr_success == max_success:

                print(f'\n >> Max success, Saving model & Obtaining new data : {ctr_successful_datasets+1} \n')

                ctr_save_id +=1
                save_id = model_id + ctr_save_id * 0.001
                res.save_model(model, save_id)
                if permanent_metadata: trainer.save_accugrads(accu_grads, save_id)
                print(f'Model and Accugrad .pkl(s) saved : {save_id}')

                if reducing_batch_sizes and ctr_save_id % reduce_batch_per_save == 0:
                    trainer.batch_size = int(trainer.batch_size * 4/5)

                ctr_successful_datasets +=1
                ctr_success = 0

                data = res.load_data(data_path, data_size) # with new data, comes new stuff.

                checkpoints[-1][-1] = loss_initial
                branches[-1][-1]    = loss_initial

                if not permanent_metadata:
                    checkpoints[-1][1] = trainer.init_accugrads(model)
                    branches[-1][1]    = trainer.init_accugrads(model)

        else:

            delta = current_loss - prev_loss

            if all(delta < max_loss_inc_delta):

                print(f'\n >> Tolerable loss detected : {ctr_loss_increase_below_delta+1}/{max_loss_inc_below_delta} \n')

                if ctr_loss_increase_below_delta < max_loss_inc_below_delta:

                    print(f'\n >> Branch created. \n')

                    ctr_loss_increase_below_delta +=1

                    res.write_loss(loss_epochs[0])

                    branches.append([model, accu_grads, current_loss])

                else:

                    print(f'\n >> Reverting to checkpoint.')

                    ctr_loss_increase_below_delta = 0

                    branches[-1] = checkpoints[-1]

            else:

                print(f'\n >> Intolerable loss detected : {ctr_loss_increase_above_delta+1}/{max_loss_inc_above_delta} \n')

                if ctr_loss_increase_above_delta < max_loss_inc_above_delta:

                    print(f'\n >> Reverting to checkpoint.\n')

                    ctr_loss_increase_above_delta +=1

                    branches[-1] = checkpoints[-1]

                else:

                    print(f'\n >> Reverting to checkpoint & Obtaining new data : {ctr_branch_n_data_reset+1}/{max_branch_n_data_reset} \n')

                    ctr_branch_n_data_reset +=1
                    ctr_success = 0
                    ctr_loss_increase_above_delta = 0
                    ctr_loss_increase_below_delta = 0

                    branches[-1] = checkpoints[-1]

                    data = res.load_data(data_path, data_size) # with new data, comes new stuff.

                    checkpoints[-1][-1] = loss_initial
                    branches[-1][-1]    = loss_initial

                    if not permanent_metadata:
                        checkpoints[-1][1] = trainer.init_accugrads(model)
                        branches[-1][1]    = trainer.init_accugrads(model)

                    if ctr_branch_n_data_reset == max_branch_n_data_reset:

                        print(f'\n>> Switching trainers.\n')

                        ctr_branch_n_data_reset = 0

                        new_checkpoints, new_branches = advance_parenting(model, accu_grads, _, model_id)

                        checkpoints.extend(new_checkpoints) ; branches.extend(new_branches)





def advance_parenting(model, accu_grads, moments, model_id):

    # initial condition(s) - per model

    ctr_successful_datasets = 0
    ctr_save_id = 0

    ctr_epoch = 1
    ctr_success = 0
    ctr_loss_increase_below_delta = 0
    ctr_loss_increase_above_delta = 0
    ctr_branch_n_data_reset = 0

    checkpoints = []
    branches = []

    data = res.load_data(data_path, data_size)
    train_fn = train_fns[secondary_trainer]
    trainer.learning_rate = learning_rate_2

    # set the first checkpoints

    checkpoints.append([model, accu_grads, moments, ctr_epoch, loss_initial])
    branches.append([model, accu_grads, moments, ctr_epoch, loss_initial])

    # start parenting

    while True:

        model, accu_grads, moments, ctr_epoch, prev_loss = branches[-1]

        model, accu_grads, moments, loss_epochs = train_fn(model, accu_grads, moments, data, ctr_epoch) ; ctr_epoch +=1

        current_loss = np.array(loss_epochs[0][0:-1])

        if all(current_loss < prev_loss):

            print(f'\n >> Epoch successful, Checkpoint created : {ctr_success+1}/{max_success} \n')

            ctr_success +=1
            ctr_loss_increase_below_delta = 0
            ctr_loss_increase_above_delta = 0

            res.write_loss(loss_epochs[0])
            real_losses[model_id].append(current_loss)

            checkpoints.append([model, accu_grads, moments, ctr_epoch, current_loss])
            branches.append([model, accu_grads, moments, ctr_epoch, current_loss])

            if ctr_success == max_success:

                print(f'\n >> Max success, Saving model & Obtaining new data : {ctr_successful_datasets+1} \n')

                ctr_save_id +=1
                save_id = model_id + ctr_save_id * 0.001
                res.save_model(model, save_id)
                if permanent_metadata:
                    trainer.save_accugrads(accu_grads, save_id)
                    trainer.save_moments(moments, save_id)
                print(f'Model and Metadata .pkl(s) saved : {save_id}')

                if reducing_batch_sizes and ctr_save_id % reduce_batch_per_save == 0:
                    trainer.batch_size = int(trainer.batch_size * 4/5)

                ctr_successful_datasets +=1
                ctr_success = 0

                data = res.load_data(data_path, data_size) # with new data, comes new stuff.

                checkpoints[-1][4] = loss_initial
                branches[-1][4]    = loss_initial

                if not permanent_metadata:
                    checkpoints[-1][1]  = trainer.init_accugrads(model)
                    branches[-1][1]     = trainer.init_accugrads(model)
                    checkpoints[-1][2]  = trainer.init_moments(model)
                    branches[-1][2]     = trainer.init_moments(model)
                    checkpoints[-1][-2] = 1
                    branches[-1][-2]    = 1

        else:

            delta = current_loss - prev_loss

            if all(delta < max_loss_inc_delta):

                print(f'\n >> Tolerable loss detected : {ctr_loss_increase_below_delta+1}/{max_loss_inc_below_delta} \n')

                if ctr_loss_increase_below_delta < max_loss_inc_below_delta:

                    print(f'\n >> Branch created. \n')

                    ctr_loss_increase_below_delta +=1

                    res.write_loss(loss_epochs[0])

                    branches.append([model, accu_grads, moments, ctr_epoch, current_loss])

                else:

                    print(f'\n >> Reverting to checkpoint.')

                    ctr_loss_increase_below_delta = 0

                    branches[-1] = checkpoints[-1]

            else:

                print(f'\n >> Intolerable loss detected : {ctr_loss_increase_above_delta+1}/{max_loss_inc_above_delta} \n')

                if ctr_loss_increase_above_delta < max_loss_inc_above_delta:

                    print(f'\n >> Reverting to checkpoint.\n')

                    ctr_loss_increase_above_delta +=1

                    branches[-1] = checkpoints[-1]

                else:

                    print(f'\n >> Reverting to checkpoint & Obtaining new data : {ctr_branch_n_data_reset+1}/{max_branch_n_data_reset} \n')

                    ctr_branch_n_data_reset +=1
                    ctr_success = 0
                    ctr_loss_increase_above_delta = 0
                    ctr_loss_increase_below_delta = 0

                    branches[-1] = checkpoints[-1]

                    data = res.load_data(data_path, data_size) # with new data, comes new stuff.

                    checkpoints[-1][2] = loss_initial
                    branches[-1][2]    = loss_initial

                    if not permanent_metadata:
                        checkpoints[-1][1]  = trainer.init_accugrads(model)
                        branches[-1][1]     = trainer.init_accugrads(model)
                        checkpoints[-1][2]  = trainer.init_moments(model)
                        branches[-1][2]     = trainer.init_moments(model)
                        checkpoints[-1][-2] = 1
                        branches[-1][-2]    = 1

                    if ctr_branch_n_data_reset == max_branch_n_data_reset:

                        print(f'\n>> Switching trainers.\n')

                        return checkpoints, branches


# helpers


def load_model_data(id):
    model = res.load_model(id)
    if model is None:
        IOdims = res.vocab_size
        architecture = model_architectures[id]
        model = Vanilla.create_model(IOdims, architecture, IOdims)
    accu_grads = trainer.load_accugrads(model, id)
    moments = trainer.load_moments(model, id)
    return model, accu_grads, moments

# def iterate_train_fns():
#     this_fn = train_fns.pop(0)
#     train_fns.append(this_fn)
#     return train_fns[0]


from multiprocessing import cpu_count
print(f'> Parent initiated. \n'
      '-----\n'
      f'Primary   : {train_fns[primary_trainer]}\n'
      f'Secondary : {train_fns[secondary_trainer]}\n'
      '-----\n'
      f'Detected CPU(s): {cpu_count()}\n'
      '-----\n'
      f'Learning Rates: {learning_rate_1} {learning_rate_2}\n'
      f'Sample Amount: {data_size}\n'
      f'Batch Size: {batch_size}\n'
      '-----')
#       'Model Architecture(s): ', end='')
# [print(_,' ', end='') for _ in model_architectures]
# print('\n-----\n')



if __name__ == '__main__':

    trainer.epochs = 1
    trainer.batch_size = batch_size

    torch.set_default_tensor_type('torch.FloatTensor')

    # with ThreadPool(multiprocessing.cpu_count()) as pool:     # if multi-model training
    #     pool.map_async(start_parenting, range(hm_models))     # use f'{model_id}-{ctr}'
    #
    #     pool.close()
    #     pool.join()

    id = 0

    if not start_advanced: start_parenting(id)
    else:
        model, accu_grads, moments = load_model_data(id)
        advance_parenting(model, accu_grads, moments, id)


    print('parent exited.')


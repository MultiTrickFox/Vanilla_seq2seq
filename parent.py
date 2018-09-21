import Vanilla
import trainer
import res
import utils

import time
import torch
import numpy as np


    # parent details

total_epochs = 20
learning_rate_1 = 0.001
learning_rate_2 = 0.01


    # model details

default_layers = [5, 8, 6]


    # data details

data_path = 'samples.pkl'
data_size = 10_000
batch_size = 500


    # training details

start_advanced = False

further_parenting = False

trainer.drop_in  = 0.0
trainer.drop_mid = 0.0
trainer.drop_out = 0.0

reducing_batch_sizes = True
reduce_batch_per_epoch = 5

save_intermediate_model = True
save_model_per_epoch = 5

branch_ctr_max = 5


    # global declarations

loss_initial = \
    [[999_999_999,999_999_999,999_999_999,999_999_999]]



def simple_parenting(model, accugrads, data):


        # initial conditions

    trainer.learning_rate = learning_rate_1

    ctr_save_id = 0

    successful_epochs = 0

    checkpoints = []

    prevStep = (model, accugrads, loss_initial)


        # begin parenting

    while successful_epochs < total_epochs:

        prev_model, prev_accugrads, prev_loss = prevStep

        thisStep = trainer.train_rms(prev_model, prev_accugrads, data) ; this_loss = thisStep[-1]

        if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

            checkpoints.append(prevStep)

            successful_epochs +=1

            print(f'@ {get_clock()} : '
                  f'epoch {successful_epochs} / {total_epochs} completed. ')
            res.write_loss(prevStep[-1][0], as_txt=True, epoch_nr=successful_epochs)

            if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                trainer.batch_size = int(trainer.batch_size * 4/5)
                print(f'Batch size reduced : {trainer.batch_size}')

            if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                res.save_model(prevStep[0], save_id)
                trainer.save_accugrads(prevStep[1], save_id)
                print(f'Data saved : {ctr_save_id}')

            prevStep = thisStep

        else:

            branch_ctr = 0

            branch_points = []

            branch_prevStep = thisStep

            branch_goal = prevStep[-1]

            while branch_ctr < branch_ctr_max:

                prev_model, prev_accugrads, prev_loss = branch_prevStep

                branch_thisStep = trainer.train_rms(prev_model, prev_accugrads, data) ; this_loss = branch_thisStep[-1]

                if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

                    branch_points.append(branch_prevStep)

                    if all(np.array(this_loss[0]) < np.array(branch_goal[0])):

                        checkpoints.append(branch_points[-1])

                        successful_epochs +=1

                        print(f'@ {get_clock()} : '
                              f'epoch {successful_epochs} / {total_epochs} completed. ')
                        res.write_loss(branch_prevStep[-1][0], as_txt=True, epoch_nr=successful_epochs)

                        if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                            trainer.batch_size = int(trainer.batch_size * 4/5)

                        if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                            ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                            res.save_model(branch_prevStep[0], save_id)
                            trainer.save_accugrads(branch_prevStep[1], save_id)

                        break

                    branch_prevStep = branch_thisStep

                branch_ctr += 1

                print(f'@ {get_clock()} : '
                      f'Branch {branch_ctr} / {branch_ctr_max} generated. ')


    del checkpoints[0]
    return checkpoints



def advanced_parenting(model, accugrads, moments, data):


        # initial conditions

    trainer.learning_rate = learning_rate_2

    ctr_save_id = 0

    successful_epochs = 0

    checkpoints = []

    prevStep = (model, accugrads, moments, loss_initial)


        # begin parenting

    while successful_epochs < total_epochs:

        prev_model, prev_accugrads, prev_moments, prev_loss = prevStep

        thisStep = trainer.train_rmsadv(prev_model, prev_accugrads, prev_moments, data) ; this_loss = thisStep[-1]

        if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

            checkpoints.append(prevStep)

            successful_epochs +=1

            print(f'@ {get_clock()} : '
                  f'epoch {successful_epochs} / {total_epochs} completed.')
            res.write_loss(prevStep[-1][0], as_txt=True, epoch_nr=successful_epochs)

            if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                trainer.batch_size = int(trainer.batch_size * 4/5)

            if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                res.save_model(prevStep[0], save_id)
                trainer.save_accugrads(prevStep[1], save_id)
                trainer.save_moments(prevStep[2], save_id)

            prevStep = thisStep

        else:

            branch_ctr = 0

            branch_points = []

            branch_prevStep = thisStep

            branch_goal = prevStep[-1]

            while branch_ctr < branch_ctr_max:

                prev_model, prev_accugrads, prev_moments, prev_loss = branch_prevStep

                branch_thisStep = trainer.train_rms(prev_model, prev_accugrads, prev_moments, data) ; this_loss = branch_thisStep[-1]

                if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

                    branch_points.append(branch_prevStep)

                    if all(np.array(this_loss[0]) < np.array(branch_goal[0])):

                        checkpoints.append(branch_points[-1])

                        successful_epochs +=1

                        print(f'@ {get_clock()} : '
                              f'epoch {successful_epochs} / {total_epochs} completed. ')
                        res.write_loss(branch_prevStep[-1][0], as_txt=True, epoch_nr=successful_epochs)

                        if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                            trainer.batch_size = int(trainer.batch_size * 4/5)

                        if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                            ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                            res.save_model(branch_prevStep[0], save_id)
                            trainer.save_accugrads(branch_prevStep[1], save_id)
                            trainer.save_moments(branch_prevStep[2], save_id)

                        break

                    branch_prevStep = branch_thisStep

                branch_ctr +=1

                print(f'@ {get_clock()} : '
                      f'Branch {branch_ctr} / {branch_ctr_max} geenrated. ')


    del checkpoints[0]
    return checkpoints



# helpers

def get_data(): return res.load_data(data_path, data_size)

def get_clock(): return time.asctime(time.localtime(time.time())).split(' ')[3]



# parent runners


def run_simple_parenting(data):

    # initialize model
    IOdims = res.vocab_size
    model = res.load_model()
    if model is None: model = Vanilla.create_model(IOdims, default_layers, IOdims)

    # initialize metadata
    accugrads = trainer.load_accugrads(model)

    # get checkpoints
    checkpoints = simple_parenting(model, accugrads, data)

    # extract metadata
    model = checkpoints[-1][0]
    accugrads = checkpoints[-1][1]
    # losses = [list(cp[-1]) for cp in checkpoints]

    # save metadata
    res.save_model(model)
    trainer.save_accugrads(accugrads)
    # [res.write_loss(loss, as_txt=True, epoch_nr=_) for _, loss in enumerate(losses)]


def run_advanced_parenting(data):

    # initialize model
    IOdims = res.vocab_size
    model = res.load_model()
    if model is None: model = Vanilla.create_model(IOdims, default_layers, IOdims)

    # initalize metadata
    accugrads = trainer.load_accugrads(model)
    moments = trainer.load_moments(model)

    # get checkpoints
    checkpoints = advanced_parenting(model, accugrads, moments, data)

    # extract metadata
    model = checkpoints[-1][0]
    accugrads = checkpoints[-1][1]
    moments = checkpoints[-1][2]
    # losses = [list(cp[-1]) for cp in checkpoints]

    # save metadata
    res.save_model(model)
    trainer.save_accugrads(accugrads)
    trainer.save_moments(moments)
    # [res.write_loss(loss, as_txt=True, epoch_nr=_) for _, loss in enumerate(losses)]



if __name__ == '__main__':

    res.initialize_loss_txt() ; torch.set_default_tensor_type('torch.FloatTensor')

    data = get_data()

    trainer.batch_size = batch_size

    if not start_advanced:

        run_simple_parenting(data)

        if further_parenting:

            run_advanced_parenting(data)

    else: # start advanced

        run_advanced_parenting(data)

    utils.plot_loss_txts()





def parent_bootstrap(hm_data,
                     batches_of,
                     total_ep=20,
                     lr_1=0.001,
                     lr_2=0.01,
                     structure=(3, 5, 4),
                     begin_advanced=False,
                     extra_care=False,
                     decay_batch_sizes=True,
                     quicksaves = True):

    global total_epochs 
    total_epochs = total_ep
    global data_size 
    data_size = hm_data
    global batch_size 
    batch_size = batches_of
    global learning_rate_1 
    learning_rate_1 = lr_1
    global learning_rate_2 
    learning_rate_2 = lr_2
    global layers 
    layers = structure
    global start_advanced 
    start_advanced = begin_advanced
    global further_parenting 
    further_parenting = extra_care
    global reducing_batch_sizes 
    reducing_batch_sizes = decay_batch_sizes
    global save_intermediate_model 
    save_intermediate_model = quicksaves

    trainer.batch_size = batch_size
    res.initialize_loss_txt()
    data = get_data()

    if not start_advanced:
        run_simple_parenting(data)

        if further_parenting: run_advanced_parenting(data)

    else: run_advanced_parenting(data)

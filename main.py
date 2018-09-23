import res
import parent
import interact

is_preprocessed = True
parent.shutdown_after_complete = False

len_data = None

if __name__ == '__main__':
    if not is_preprocessed:
        len_data = res.preprocess_bootstrap()
    if len_data is None:
        len_data = input('Enter Data Amount.. ')
        # todo: read from pickle here
    parent.parent_bootstrap(len_data, parent.batch_size)
    interact.bootstrap()

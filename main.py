import res
import parent
import interact

is_preprocessed = False # todo: res.is_preprocessed()

len_data = None

if __name__ == '__main__':
    if not is_preprocessed:
        len_data = res.preprocess_bootstrap()
    if len_data is None:
        len_data = res.get_datasize(parent.data_path)
    parent.parent_bootstrap(len_data, parent.batch_size)
    # interact.bootstrap()

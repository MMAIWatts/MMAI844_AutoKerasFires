from google.api_core.exceptions import NotFound
from pathlib import Path
from blob_utils import *

BUCKET_NAME = 'citric-inkwell-268501'


def load_datasets():
    """
    Loads a pre-split, pre-scaled dataset from cloud to local disk

    :returns: training sets, testing sets as lists
    """

    # Check for existance of local model_cache and create if it does not exist
    if os.path.isdir('./model_cache'):
        print('model_cache exists')
    else:
        os.makedirs('./model_cache')
        print('Created model_cache directory')

    # Check existence of train_data and create if it does not exist
    if os.path.isdir('./model_cache/train_data'):
        print('train_data path exists')
    else:
        os.makedirs('./model_cache/train_data')
        print('Created train_data directory')

    # Check existence of test_data and create if it does not exist
    if os.path.isdir('./model_cache/test_data'):
        print('test_data path exists')
    else:
        os.makedirs('./model_cache/test_data')
        print('Created test_data directory')

    bucket_files = ['training_sets/full_augmentation/full_augmentation_train_x_aug.npy',
                    'training_sets/full_augmentation/full_augmentation_train_y_aug.npy',
                    'test_set/test_x.npy',
                    'test_set/test_y.npy']

    training_sets = []
    testing_sets = []
    for dataset in bucket_files:
        p = Path(dataset)

        # Check whether the file is a training or testing dataset
        if 'training_sets' in dataset:
            data_type = 'train_data'
            training_sets.append(dataset.replace('/', '-'))
        elif 'test_set' in dataset:
            data_type = 'test_data'
            testing_sets.append(dataset.replace('/', '-'))
        else:
            continue

        # Check if the file is already downloaded and download it if not
        if os.path.exists(os.path.join(f'./model_cache/{data_type}', str(dataset.replace('/', '-')))):
            print(f'{p.name} already downloaded')
        else:
            print(f'{p.name}  downloading...')
            try:
                download_blob(BUCKET_NAME, dataset, os.path.join(f'./model_cache/{data_type}', str(dataset.replace('/', '-'))))
                print(f'{p.name}  done downloading')
            except NotFound:
                print(f'File {dataset} not found in cloud storage blob.')
                # Remove empty file
                os.remove(os.path.join(f'./model_cache/{data_type}', str(dataset.replace('/', '-'))))

    return training_sets, testing_sets


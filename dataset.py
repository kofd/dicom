from os import path, listdir
import numpy as np

from util import intchars, imap_async, ibatch
from config import DATA_DIR
from sample import DicomDatum, sample_datum, batch_fn


def assemble_annotation_data():
    '''
    Assembles all of the data stored in the data directory into annotations that can
    be used to retrieve all of the data for a given sample.

    :return: a list of dictionaries denoting the location of the data relevant to
             constructing a training sample.
    '''
    data = []
    with open(path.join(DATA_DIR, 'link.csv')) as f:
        patients = [line.strip().split(',') for line in f.readlines()[1:]]
    for dicom, contour in patients:
        i_contours = set(listdir(path.join(DATA_DIR, 'contourfiles', contour, 'i-contours')))
        o_contours = set(listdir(path.join(DATA_DIR, 'contourfiles', contour, 'o-contours')))
        for dfilename in listdir(path.join(DATA_DIR, 'dicoms', dicom)):
            index = int(dfilename.rstrip('.dcm'))
            
            i_contour_name = 'IM-0001-{}-icontour-manual.txt'.format(intchars(index, 4))
            if i_contour_name in i_contours:
                i_contour = '{}/i-contours/{}'.format(contour, i_contour_name)
            else:
                i_contour = None
                
            o_contour_name = 'IM-0001-{}-ocontour-manual.txt'.format(intchars(index, 4))
            if o_contour_name in o_contours:
                o_contour = '{}/o-contours/{}'.format(contour, o_contour_name)
            else:
                o_contour = None
                
            data.append(DicomDatum(
                dicom='{}/{}'.format(dicom, dfilename),
                i_contour=i_contour,
                o_contour=o_contour
            ))
    return data


class DicomDataset:
    def __init__(self):
        self.data = assemble_annotation_data()

    def train_epoch(self, batch_size=8):
        '''
        Returns a generator of batches of training data over a single epoch.

        :param batch_size: Batch size of the batches.
        :return: a generator yielding dictionaries of numpy arrays with the 0 axis
                 representing the batch.
        '''
        epoch = list(self.data)
        np.random.shuffle(epoch)
        # sample function returns None if there is no data for datum.
        stream = (value for value in imap_async(sample_datum, epoch) if value is not None)
        yield from ibatch(stream, batch_size, batch_fn)


def verify_assembled_data():
    import json

    data = assemble_annotation_data()
    print('total datums parsed:', len(data))
    assert len(data) > 0

    target_contour = 'SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt'
    target_dicom = 'SCD0000101/48.dcm'

    for datum in data:
        if datum.dicom == target_dicom and datum.i_contour == target_contour:
            print('target datum:', json.dumps({
                'dicom': datum.dicom,
                'i_contour': datum.i_contour,
                'o_contour': datum.o_contour
            }, indent=2))
            break
    else:
        assert False, 'could not find target datum in dataset.'

    '''
    expected output:
    total datums parsed: 1140
    target datum: {
      "dicom": "SCD0000101/48.dcm",
      "i_contour": "SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt",
      "o_contour": null
    }
    '''


def verify_sampling_function():
    data = assemble_annotation_data()
    samples = [value for value in imap_async(sample_datum, data)]
    print('total resultant samples:', len(samples))
    assert len(data) == len(samples)

    valid_samples = [value for value in samples if value is not None]
    print('total valid datums:', len(valid_samples))
    assert len(samples) == len(valid_samples)

    '''
    expected output:
    total resultant samples: 1140
    total valid datums: 1140
    '''


def verify_dataset():
    dataset = DicomDataset()
    print('expected batches per epoch:', len(dataset.data) // 8)

    epoch = list(dataset.train_epoch(8))
    print('actual batches per epoch:', len(epoch))
    assert len(dataset.data) // 8 == len(epoch)

    images, targets = epoch[0]
    assert images.shape == (8,256,256,1)
    assert images.dtype == np.int16
    assert targets.shape == (8,256,256,2)
    assert targets.dtype == bool

    '''
    expected output:
    expected batches per epoch: 142
    actual batches per epoch: 142
    '''


if __name__ == '__main__':
    verify_assembled_data()
    verify_sampling_function()
    verify_dataset()

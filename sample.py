from os import path
import numpy as np

from parsing import parse_dicom_file, parse_contour_file, poly_to_mask
from config import DATA_DIR


class DicomDatum:
    def __init__(self, dicom, i_contour, o_contour):
        self.dicom = dicom
        self.i_contour = i_contour
        self.o_contour = o_contour

    @property
    def dicom_path(self):
        return path.join(DATA_DIR, 'dicoms', self.dicom)

    @property
    def i_contour_path(self):
        return path.join(DATA_DIR, 'contourfiles', self.i_contour)

    @property
    def o_contour_path(self):
        return path.join(DATA_DIR, 'contourfiles', self.o_contour)


def sample_datum(spec):
    '''
    Given a contour datum, open the files associated, parse them and return training
    data as a dictionary of numpy arrays.

    :param spec: A ContourDatum object.
    :return: A dictionary of numpy arrays representing the training data for the datum.
    '''
    dicom = parse_dicom_file(spec.dicom_path)
    if dicom is None:
        return None
    dicom = dicom['pixel_data']

    if spec.i_contour is not None:
        contours = parse_contour_file(spec.i_contour_path)
        i_contour = poly_to_mask(contours, *dicom.shape)
    else:
        i_contour = np.zeros(dicom.shape, dtype=bool)
    i_contour = np.expand_dims(i_contour, -1)

    if spec.o_contour is not None:
        contours = parse_contour_file(spec.o_contour_path)
        o_contour = poly_to_mask(contours, *dicom.shape)
    else:
        o_contour = np.zeros(dicom.shape, dtype=bool)
    o_contour = np.expand_dims(o_contour, -1)

    return {
        'image': np.expand_dims(dicom, -1),
        'target': np.concatenate((i_contour, o_contour), -1)
    }


def batch_fn(btc):
    '''
    Assemble a batch of dictionaries of numpy arrays into a tuple representing the
    data for dicoms and contours into a tuple of batches.

    :param btc: A batch of dictionaries of numpy arrays.
    :return: A tuple representing the dicom and the contour batches.
    '''
    dic = {key: np.array([datum[key] for datum in btc]) for key in btc[0].keys()}
    return dic['image'], dic['target']

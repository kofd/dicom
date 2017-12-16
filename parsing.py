"""Parsing code for DICOMS and contour files"""

import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw


def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))

    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        dcm_dict = {'pixel_data' : dcm_image}
        return dcm_dict
    except InvalidDicomError:
        return None


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask


def verify_parsing():
    from os import path

    from config import DATA_DIR

    contour_filename = path.join(DATA_DIR, 'contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt')
    dicom_filename = path.join(DATA_DIR, 'dicoms/SCD0000101/48.dcm')

    print('loading contour file:', contour_filename)
    contours = parse_contour_file(contour_filename)
    print('contours')
    for line in contours[:3]:
        print(line)
    print('...')
    for line in contours[-3:]:
        print(line)

    assert len(contours[0]) == 2

    print('loading dicom file:', dicom_filename)
    dicom_data = parse_dicom_file(dicom_filename)
    print('dicom shape:', dicom_data['pixel_data'].shape)
    assert dicom_data['pixel_data'].shape == (256,256)
    Image.fromarray(dicom_data['pixel_data']).show()

    mask = poly_to_mask(contours, *dicom_data['pixel_data'].shape)
    print('mask shape, min, max:', mask.shape, np.min(mask.astype(float)), np.max(mask.astype(float)))
    Image.fromarray(mask.astype(float) * 255).show()
    assert mask.shape == dicom_data['pixel_data'].shape
    assert mask.dtype == bool

    '''
    expected output:
    loading contour file: .../data/final_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt
    contours
    (120.5, 137.5)
    (120.5, 137.0)
    (121.0, 136.5)
    ...
    (120.5, 139.0)
    (120.5, 138.5)
    (120.5, 138.0)
    loading dicom file: .../data/final_data/dicoms/SCD0000101/48.dcm
    dicom shape: (256, 256)
    mask shape, min, max: (256, 256) 0.0 1.0
    '''


if __name__ == '__main__':
    verify_parsing()

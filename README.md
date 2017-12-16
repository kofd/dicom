DICOM training set library
-

This library is first pass code at training a model on dicom images.

It is currently organized as a series of scripts in the root directory but is intended to eventually be folded into its
own library. 'parsing.py' contains a tool to read dicom images into numpy objects, 'sample.py' contains code to supply
the results of those operations in a training pipeline environment, and 'dataset.py' offers a tool to batch iterate
through epochs of data.

To verify the parsing system, it can be executed directly or call parsing.verify_parsing(). It should parse specific
datums in the dataset, verify the results fit the desired schema and offer visualizations of the resulting data so that
the label can be inspected against the input.

To verify the sampling system, several verification functions can be found in 'dataset.py' and can be executed directly
from 'dataset.py'. It will search through the local dataset, inspect the results of collecting annotations, async 
iterating over the annotations to construct batches of sampled data, and finally initializing a dataset object that can
then be used to construct epochs of training data.

Improvemnts to be made:
-
- Add a validation set to the dataset class.
- Integrate an actual model into the training flow.
- Add additional logging and error handling to the sampling system
- Add weighted sampling or include sample weight channel in sampling.
- Add code to dynamically download the dataset if it hasn't been cached on disk yet.
- Assemble batches in a separate thread so it doesn't block the gpu.
- If the dataset is provably small, keep entire sampled dataset in memory for performance reasons.
- Add data augmentation.

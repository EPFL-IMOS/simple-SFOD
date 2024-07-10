import numpy as np
from netcal.scaling import LogisticCalibration, LogisticCalibrationDependent

#matched                # binary NumPy 1-D array (0, 1) that indicates if a bounding box has matched a ground truth at a certain IoU with the right label - shape: (n_samples,)
#confidences            # NumPy 1-D array with confidence estimates between 0-1 - shape: (n_samples,)
#relative_x_position    # NumPy 1-D array with relative center-x position between 0-1 of each prediction - shape: (n_samples,)

matched = np.array([1,0,1,1,0])
confidences = np.array([0.8, 0.9, 0.7, 0.6, 0.5])
# confidences = np.array([1,0,1,1,0])
relative_x_position = np.array([0, 0.2, 0.4, 0.6, 0.8])
#relative_x_position = np.array([0,0,0,0,0])

input = np.stack((confidences, relative_x_position), axis=1)
# print(input)

lr = LogisticCalibration(detection=True, use_cuda=False)    # flag 'detection=True' is mandatory for this method
lr.fit(input, matched)
calibrated = lr.transform(input)

lr_dependent = LogisticCalibrationDependent(use_cuda=False) # flag 'detection=True' is not necessary as this method is only defined for detection
lr_dependent.fit(input, matched)
calibrated = lr_dependent.transform(input)

from netcal.metrics import ECE

n_bins = 10
input_calibrated = np.stack((calibrated, relative_x_position), axis=1)

ece = ECE(n_bins, detection=True)           # flag 'detection=True' is mandatory for this method
confidences = confidences.reshape(-1, 1)
uncalibrated_score = ece.measure(confidences, matched)
calibrated_score = ece.measure(input_calibrated, matched)
print(input)
print(input_calibrated)
print(uncalibrated_score)
print(calibrated_score)

from netcal.presentation import ReliabilityDiagram

n_bins = [10, 10]

diagram = ReliabilityDiagram(n_bins, detection=True)    # flag 'detection=True' is mandatory for this method
diagram.plot(input, matched)                # visualize miscalibration of uncalibrated
diagram.plot(input_calibrated, matched)     # visualize miscalibration of calibrated

# you can also use this method to create a tikz file with tikz code
# that can be directly used within LaTeX documents:
diagram.plot(input, matched, tikz=True, filename="diagram.tikz")
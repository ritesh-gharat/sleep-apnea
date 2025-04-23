import wfdb
import matplotlib.pyplot as plt

# Load the ECG signal and header
record = wfdb.rdrecord("./data/a01")

# Plot ECG signal
wfdb.plot_wfdb(record=record, title="ECG Signal")

# Load apnea annotations
annotations = wfdb.rdann("./data/a01", "apn")

# Print apnea annotations
print("Apnea Annotations:")
print(annotations.sample)  # Indices where annotations occur
print(annotations.symbol)  # Corresponding apnea labels

# Load QRS annotations
qrs_annotations = wfdb.rdann("./data/a01", "qrs")

# Print QRS annotation details
print("QRS Annotations:")
print(qrs_annotations.sample)  # QRS detection points

# Plot ECG with QRS annotations
wfdb.plot_wfdb(record=record, ann=qrs_annotations, title="ECG with QRS Annotations")

## Environment Dependencies

Before running the project scripts, please install the required Python packages, including cheminformatics, deep learning, machine learning libraries, and basic packages for plotting and mathematical calculation:

- **Core libraries**: rdkit, pytorch, scikit\-learn

- **Basic auxiliary libraries**: Common Python packages for data visualization and mathematical computation \(e\.g\., numpy, matplotlib, pandas\)

## Script Instructions \& Running Rules

This project contains three core Python scripts with independent functions and fixed running requirements\. Please follow the specified running sequence and times strictly\.

### 1\. MolecularFingerprint\.py

This script is used to generate molecular fingerprint features for 14 types of solvents\. **It only needs to be executed once** in the whole project running process, and no repeated operation is required\.

### 2\. RFE_FeatureSelection\.py

This script implements the Recursive Feature Elimination \(RFE\) algorithm for feature screening\. It only adopts pre\-experimental data for calculation and analysis\. **It only needs to be run once**\.

### 3\. RandomForest\.py

This is the core script for model evaluation and prediction in the project\. It will read experimental results of all previous batches, construct and train the prediction model based on the imported data, and finally output the prediction results of the next batch of experiments\. This script can be executed multiple times according to experimental iteration demands\.

### 4\. Note
MolecularFingerprint\.py contains the codes for formula generation of the next batch of experiments. Due to different formulation‑generation strategies adopted in each iteration loop, and random perturbations introduced during solvent ratio generation, this part has not been organized as a standalone runnable Python program.

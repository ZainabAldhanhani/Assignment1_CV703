# Assignment1 CV703

## Repository Structure

- `Flowers102_ImageClassification/`: Image classification methods on Flowers102 dataset
- `Imagewoof_ImageClassification/`: Image classification methods on Imagewoof dataset
- `combinedSet_ImageClassification/`:Image classification methods on a combined set of (Flowers102 + FGVC Aircraft + Imagewoof)
- `Jupyter Notebook/`: Assignment1 notebook
- `README.md`: Overview and setup instructions.
- `requirements.txt`: Required libraries for the Assignment.


## Architecture 
Our modified ConvNeXt architecture improves performance and generalization through several enhancements. We applied stronger data augmentations like random flips, rotations, color jitter, and MixUp to improve robustness. The classifier head was updated with Dropout for better regularization, and AdamW replaced Adam for improved optimization. We also switched to SoftTargetCrossEntropy for better soft-label training and implemented a learning rate scheduler to enhance stability and convergence. These modifications collectively boost model performance and reduce overfitting.
<img src="Figures/Diagram.png" alt="Diagram" width="400">

## Dsataset 
<img src="Figures/Flowers102.png" alt="Flowers102" width="400">

<img src="Figures/Imagewoof.png" alt="Flowers102" width="400">

<img src="Figures/combinedSet.png" alt="Flowers102" width="400">


## Install Requirements
Clone this repository and install the required Python packages:

```bash
git clone https://github.com/ZainabAldhanhani/Assignment1_CV703.git
cd Assignment1_CV703
pip install -r requirements.txt
```
## Train and Evaluate ConvNeXt on Flowers102 dataset
#### ConvNeXt
```bash
cd Flowers102_ImageClassification
python Flowers102_ConvNeXt.py
```
#### The proposed solution (Modified ConvNeXt)
```bash
cd Flowers102_ImageClassification
python Flowers102_ModifiedConvNeXtTiny.py
```

## Train and Evaluate ConvNeXt on Imagewoof dataset
#### ConvNeXt
```bash
cd Imagewoof_ImageClassification
python Imagewoof_ConvNeXt.py
```
#### The proposed solution (Modified ConvNeXt)
```bash
cd Imagewoof_ImageClassification
python Imagewoof_ModifiedConvNeXtTiny.py
``` 


## Train and Evaluate ConvNeXt on combined set of(Flowers102 + FGVC Aircraft + Imagewoof)
#### ConvNeXt
```bash
cd combinedSet_ImageClassification
python combinedSet _ConvNeXt.py
```
#### The proposed solution (Modified ConvNeXt)
```bash
cd Imagewoof_ImageClassification
python combinedSet _ModifiedConvNeXtTinyConvNeXtTiny.py
```

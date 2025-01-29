# Assignment1 CV703

## Repository Structure

- `Flowers102_ImageClassification/`: Image classification methods on Flowers102 dataset
- `Imagewoof_ImageClassification/`: Image classification methods on Imagewoof dataset
- `combinedSet_ImageClassification/`:Image classification methods on a combined set of (Flowers102 + FGVC Aircraft + Imagewoof)
- `Assignment1CV703.ipynb`: Assignment1 notebook
- `README.md`: Overview and setup instructions.
- `requirements.txt`: Required libraries for the Assignment.

## Architecture
![Diagram](Figures/Diagram.png "Diagram")

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

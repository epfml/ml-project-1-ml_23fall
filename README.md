# CS433-ML Project1 

- estimate the likelihood of developing MICHD

This project uses machine learning to predict an individual's risk of developing coronary heart disease based on lifestyle and clinical factors obtained from the BRFSS survey. The goal is to enable early detection and prevention of cardiovascular disease by identifying high-risk characteristics.

## Installation

To run this project, you will need to install the following Python packages:

### Prerequisites

Make sure you have Python installed on your system. 

### Required Packages

The project requires `numpy` and `matplotlib` to process data and generate visualizations respectively. You can install them using `pip`:

```bash
pip install numpy 
pip install matplotlib
```

## Usage

Follow these steps to  run the project:

1. Create a folder named `dataset` in the root directory of the project.

2. Unzip the `dataset.zip` file and place the `x_test`, `x_train`, and `y_train` files into the `dataset` folder.

3. Execute the `run.py` script to start the model training and evaluation process:

```bash
python run.py
```

## Project Structure

This section outlines the structure of the project and the purpose of each file:

- `run.py`: The main script，which contains the final pipeline.

- `implementations.py`: Implementations of basic methods.

- `model.py`: Includes utility functions for cross-validation and model evaluation, and adjustments for dataset format.

- `helpers.py`: Various helper functions.

- `zhyu_dataexplore.ipynb`: Exploratory analysis and feature engineering of datasets


  

  

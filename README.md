# cgpdm-lib
Library implementing __Controlled Gaussian Process Dynamical Models__ (C-GPDMs).

Class __GPDM()__ implements original model from __Wang et al. (2005) "Gaussian process dynamical models"__.

Class __CGPDM()__ extends the main class to take into account also the presence of control inputs.

## Installation

To install __cgpdm_lib__ on your system, clone the repository, open it in a terminal and run the following command:

```
pip install .
```

Instead if you want to install the package in "editable" or "develop" mode (to prevent the uninstall/install of the
package at every package change) run the following command:

```
pip install -e .
```

## Dependencies
- [PyTorch] (https://pytorch.org/)
- [NumPy] (https://numpy.org/)
- [Matplotlib] (https://matplotlib.org/)
- [Scikit-learn] (https://scikit-learn.org/stable/)

## Usage
Open a terminal inside __test/__ folder
- Run __$ python test_cgpdm.py__ to apply GPDM dimensionality reduction to state-input cloth movement data (stored inside folder __test/DATA/__).

__test_cgpdm.py__ take command line arguments:
- __seed__: select the random seed
- __num_data__: select the number of trajectories used for training
- __deg__: select the oscillation angle used in data collection (5, 10 or 15)
- __d__: select the latent space dimension
- __num_opt_steps__: select the number of optimization steps
- __lr__: select the optimization learning rate
- __flg_show__: set to 'True' for showing model results

## Citing
If you use this package, please cite the following paper: __Amadio et al. "Controlled Gaussian Process Dynamical Models with Application to Robotic Cloth Manipulation"__.

# DriVe-forecast
Code accompanying the paper "Real-Time Forecasting of Driver-Vehicle Dynamics on 3D Roads: a Deep-Learning Framework Leveraging Bayesian Optimisation" by Luca Paparusso, Stefano Melzi, and Francesco Braghin.

## Installation
### Cloning
Clone the repository and all the submodules, using:
```
git clone --recurse-submodules <repository cloning URL>
```

### Environment setup
Create conda environment and install dependencies
```
conda create --name DriVe-forecast python=3.6 -y
source activate DriVe-forecast
pip install -r requirements.txt
```

Finally, install the conda environment as a kernel, for usage in IPython notebooks:
```
python -m ipykernel install --user --name DriVe-forecast --display-name "Python 3.6 (DriVe-forecast)"
```

## Contents
- paper_plots.ipynb: Jupyter notebook replicating the paper plots;
- plotting_functions.py: plotting utilities called in paper_plots.ipynb;
- train.py: code for training the framework;
- functions.py: functions called in train.py.

## Citation
If you used this package in your research, cite it as:
<pre>
    <code>
@Misc{,
      author = {L. Paparusso and S. Melzi and F. Braghin},
      title = {{DriVe-forecast}},
      year = {2021--},
      url = " https://github.com/lpaparusso/DLfmwk_driver"
}
    </code>
</pre>

## Dependencies (defined by their "pip" name)
- numpy
- tensorflow
- keras
- scipy
- pandas
- bayesian-optimization

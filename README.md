# Cross Estimation Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code to to reproduce the results in "A Neural Network Paradigm for Modeling Psychometric Data and Estimating IRT Model Parameters: Cross Estimation Network" by Longfei Zhang and Ping Chen, which is published in *Behavior Research Methods*.

DOI 
https://doi.org/10.3758/s13428-024-02406-3


The core code for implementing the CEN approach resides in ```src/models/cen.py```.

## Instructions
To begin, run the `generator.py` script located in the `src/data` folder to generate the response data sets that are going to be used in the simulation studies.

```python
python3 generator.py
```

Next, run the following scripts in the `src/runners` folder to conduct the four simulation studies to evaluate the performance of CEN in different scenarios.

```python
python3 run_sim1.py
```
```python
python3 run_sim2.py
```
```python
python3 run_sim3.py
```
```python
python3 run_suppl_sim.py
```

To conduct the empirical study, use the following code:
```python
python3 run_emp.py
```

Additionally, visualize the trained PN and IN models with the script below.
```python
python3 vis_cen.py
```
## Acknowledgement
We are indebted to the editors and reviewers of *Behavior Research Methods* for their invaluable time and effort devoted to enhancing the quality of this work. Their insightful feedback and constructive comments have greatly contributed to the refinement of our manuscript.

## Contact
Please feel free to email [Longfei Zhang](mailto:zhanglf@mail.bnu.edu.cn?subject=[CrossEstimationNetwork]) with any questions or comments.

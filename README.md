# Adaptive trophic dismantling reveals scale-free structural avalanches in directed networks


## Repo Contents
- [Code](https://github.com/GavinHust/ATAD/tree/main/Code): The source code in the paper. 
  - [ATAD_analysis.py](https://github.com/GavinHust/ATAD/tree/main/Code/ATAD_analysis.py) is mainly used to generate intermediate results of network dismantling.
  - [ATAD_draw.py](https://github.com/GavinHust/ATAD/tree/main/Code/ATAD_draw.py) is mainly used to generate the figures in the paper based on the results of network dismantling.
- [Data](https://github.com/GavinHust/ATAD/tree/main/Data): The data in the paper mainly include the original network data, such as Synthetic network and Real network datasets, as well as the network dismantling result information data used to draw the result pictures in the paper.
  - [Networks](https://github.com/GavinHust/ATAD/tree/main/Data/Networks): The synthetic networks and real-world networks used in the paper:
    - [ER_network](https://github.com/GavinHust/ATAD/tree/main/Data/Networks/ER_network): ER networks with average degrees of 3, 6, 9, and 12. 
    - [SF_network](https://github.com/GavinHust/ATAD/tree/main/Data/Networks/SF_network): scale-free networks with power-law exponents of 2.2, 2.5, 2.8, and 3.2. 
    - [large_scale_SF_network](https://github.com/GavinHust/ATAD/tree/main/Data/Networks/large_scale_SF_network): SF networks with 10,000 nodes, used for the SOC analysis in the paper.
    - [real_network](https://github.com/GavinHust/ATAD/tree/main/Data/Networks/real_network): real-world networks used in the paper.
  - [ND_results](https://github.com/GavinHust/ATAD/tree/main/Data/ND_results): The network dismantling intermediate results of different methods used to draw the result images in the paper, including data in NPY format and CSV format.
  - [other](https://github.com/GavinHust/ATAD/tree/main/Data/other) and [p_values](https://github.com/GavinHust/ATAD/tree/main/Data/p_values): Some other related data used for conducting experiments and drawing pictures.

## Software Dependencies
Users should first install the following software packages in the virtual environment of python3.7. The version of the software, specifically, is:
```
matplotlib==3.5.1
networkx==2.6.3
numpy==1.21.6
powerlaw==1.5
scipy==1.7.3
torch==1.13.1+cu116
torch_geometric==2.3.1
```
We also provide the requirement.txt, and users can simply install it through the following command:
```
pip install -r requirements.txt
```

## Instructions to run
1. Generate the data of GSCC for each step of network dismantling using synthetic networks and real networks. You can use the code [ATAD_analysis.py](https://github.com/GavinHust/ATAD/tree/main/Code/ATAD_analysis.py) to dismantle different networks, including "SF", "ER", "large_scale_SF_network" and "real-world".
```
python ATAD_analysis.py
```
2. Draw the pictures used in the main text and supplementary information of the paper.
```
python ATAD_draw.py
```


# Reference

Please cite our work if you find our code/paper is useful to your work. 

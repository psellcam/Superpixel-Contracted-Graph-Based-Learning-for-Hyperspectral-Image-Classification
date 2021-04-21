# Superpixel-Contracted-Graph-Based-Learning-for-Hyperspectral-Image-Classification
Code for the Paper "Superpixel Contracted Graph-Based Learning for Hyperspectral Image Classification"

If you use this code please cite:
Philip Sellars, Angelica I. Aviles-Rivero, and Carola-Bibiane Schönlieb.
Superpixel contracted graph-based learning for hyperspectral image classification.
IEEE Transactions on Geoscience and Remote Sensing (2020).


Or bib format
@article{sellars2020superpixel,
  title={Superpixel contracted graph-based learning for hyperspectral image classification},
  author={Sellars, Philip and Aviles-Rivero, Angelica I and Sch{\"o}nlieb, Carola-Bibiane},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2020},
  publisher={IEEE}
}


#Instructions for Using the Code 

1. Download all the files in the GitHub

2. Create a Python Environment with the relevalant modules. From a fresh empty conda enviroment (python version 3.9.4) you will need to install:
  ----- numpy (1.19.2)
  ----- cython (0.29.23)
  ----- scipy (1.6.2)
  ----- scikit-learn (0.24.1)
  ----- numexpr (2.7.1)
  ----- matplotlib. (3.3.4)
 

3. In your terminal run "python setup.py build_ext --inplace". This will compile the three Cython files "HMSCython.pyx", "lcmr_cython.pyx" and "graph.pyx".

4. For the datasets used in the paper i.e. "Salinas" "PaviaUni" and "Indiana Pines" you can download all the files at http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes. 

5. The file "data_analysis.py" contains the loading instructions for these datasets. Please make sure to save the datasets in folders titled "Salinas"  "PaviaUni" and "Indiana" respectively. As an alternative feel free to change the loading function to something else.

6. Command line arguments selecting the dataset, the number of labelled points and other arguments are available in the file cli.py.

7. To run the program please use python main.py 

# Example terminal input
To classify Indiana Pines with 20 ground truth labels per class ten times run

> python main.py --dataset=Indiana --num-labeled=20 --run-n=10


# Python Version
This version was ran with python 3.7.9

# Problems

If you are having any problems with the code, or I have made a mistake in the readme section, please feel free to email me at ps644@cam.ac.uk

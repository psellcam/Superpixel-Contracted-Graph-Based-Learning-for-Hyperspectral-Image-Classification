# Superpixel-Contracted-Graph-Based-Learning-for-Hyperspectral-Image-Classification
Code for the Paper "Superpixel Contracted Graph-Based Learning for Hyperspectral Image Classification"

#Instructions for Using the Code 

1. Download all the files in the GitHub

2. Create a Python Environment with the relevalant modules. All modules are standard, Numpy Scipy etc. However, make sure you have the Cython module installed.

3. In your terminal run "python setup.py build_ext --inplace". This will compile the three Cython files "HMSCython.pyx", "lcmr_cython.pyx" and "graph.pyx".

4. For the datasets used in the paper i.e. "Salinas" "PaviaUni" and "Indiana Pines" you can download all the files at http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes. 

5. The file "data_analysis.py" contains the loading instructions for these datasets. Please make sure to save the datasets in folders titled "Salinas"  "PaviaUni" and "Indiana" respectively. As an alternative feel free to change the loading function to something else.

6. "main_clean.py" and "main.py" are the main files with "main_clean.py" being a cleaner written version but "main.py" contains more detailed comments.  



# Problems

If you are having any problems with the code, or I have made a mistake in the readme section, please feel free to email me at ps644@cam.ac.uk

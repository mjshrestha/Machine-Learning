## Markov Decision Processes

In this project, two types of MDP problems were solved using three different algorithms – two planning algorithms i.e, Value Iteration, Policy Iteration and one reinforcement learning algorithm i.e, Q-Learning and the results from the algorithms were compared. The two problems include one non-grid problem i.e., forest management and another grid problem - finding the route in a departmental store floor plan design. The non-grid problem was solved using Mdptoolbox [[1](https://github.com/sawcordwell/pymdptoolbox)] and the grid problem used Grid World implementation from the Brown-UMBC Reinforcement Learning and Planning Java open-source library [[2](https://github.com/jmacglashan/burlap)]

### Problems:
#### 1. Forest Problem:
1. Install Python 3.6 with following packages:
- pandas, numpy, scikit-learn, matplotlib
2. Install mdptoolbox
3. Install Jupyter Notebook
4. Run **ML_Assignment4_Forest.ipynb**

#### 2. Grid Problem:
1. Install jython.
2. Compile BURLAP source to jar file
3. Use jython to run smallGW and largeGW python files.
* *Small grid problem:*
``D:\jython2.7.1\bin\jython smallGW.py``

* *Large grid problem:*
``D:\jython2.7.1\bin\jython largeGW.py``


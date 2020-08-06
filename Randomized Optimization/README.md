# Randomized Optimization
In this project, four different randomized optimization algorithms are analyzed i.e., two instance based - Random Hill Climbing, Simulated Annealing, and two distribution based - Genetic Algorithm and MIMIC. The first part (Part1) of the assignment compares the performance of neural network backpropagation with the four randomized search algorithms on “Pima Indian Diabetes” [1](https://www.openml.org/d/37). The second part (Part2) of the assignment highlights the strength and weaknesses of the algorithms using three different optimization problems – Four peaks, Continuous Peaks and Traveling Salesman. The project uses ABAGAIL (java) [2](https://github.com/pushkar/ABAGAIL) library for the implementation of the algorithms.

## Requirements
These instructions apply for Windows 10x64. 
* Download [ABAGAIL package.](https://github.com/pushkar/ABAGAIL)  
* Download [Apache Ant](https://ant.apache.org/bindownload.cgi)
* Download [Java Development Kit](https://www.oracle.com/technetwork/java/javase/downloads/jdk10-downloads-4416644.html)
* Set path for Java and Ant. Guide [here.](https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/)


## Instructions
1. Download the preprocessed dataset file - pima_diabetes_preprocessed.csv
2. Update the location of dataset in the code.
2. Move the java files to the location ~\ABAGAIL\opt\test.
3. Follow the following instructions for ***Part 1*** and ***Part 2.***.
4. The model results (*fitness function* values and *training times*) are stored in .csv files located at ~\ABAGAIL\Results

### Part 1: Training a Neural Network using Random Search (RHC, SA, GA) and comparison with Backprop
Running the Models (via command prompt):
1. Go to Abagail folder```cd ~\ABAGAIL```
2. Compile and update jar ```ant```
3. Run RHC. ```java -cp ABAGAIL.jar opt.test.diabetes_rhc_hw2```
4. Run SA.```java -cp ABAGAIL.jar opt.test.diabetes_sa_hw2```
5. Run GA.```java -cp ABAGAIL.jar opt.test.diabetes_ga_hw2```
6. Compare all algorithms.```java -cp ABAGAIL.jar opt.test.diabetes_comparison_hw2```

### Part 2: Three Optimization Problems
1. Four Peaks Problem 
```java -cp ABAGAIL.jar opt.test.FourPeaksTest_hw2```

2. Continuous Peaks Problem 
```java -cp ABAGAIL.jar opt.test.ContinuousPeaks_hw2```

3. Traveling Salesman Problem 
```java -cp ABAGAIL.jar opt.test.TravelingSalesman_hw2```





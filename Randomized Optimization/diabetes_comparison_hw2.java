package opt.test;

/*
 * Created by Manjina Shrestha on 10/03/2019.
 * Adapted from PhishingWebsites.java by Daniel Cai (in turn adapted from AbaloneTest.java by Hannah Lau)
 */
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class diabetes_comparison_hw2 {
	private static Instance[] instances = initializeInstances();
	private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 614);
	private static Instance[] test_set = Arrays.copyOfRange(instances, 614, 768);;

	private static DataSet set = new DataSet(train_set);

	private static int inputLayer = 8, hiddenLayer1=16, hiddenLayer2=16, hiddenLayer3=16, outputLayer = 1;
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

	private static ErrorMeasure measure = new SumOfSquaresError();

	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[4]; //extra 1 for backprop
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

	private static LinearActivationFunction activation = new LinearActivationFunction();
	private static RPROPUpdateRule rule = new RPROPUpdateRule();

	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
	private static String[] oaNames = {"RHC", "SA", "GA", "Backprop"};
	private static String results = "";

	private static DecimalFormat df = new DecimalFormat("0.000");



	public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
		try {
			if (final_result) {
				String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
				String full_path = augmented_output_dir + "/" + file_name;
				Path p = Paths.get(full_path);
				if (Files.notExists(p)) {
					Files.createDirectories(p.getParent());
				}
				PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
				synchronized (pwtr) {
					pwtr.println(results);
					pwtr.close();
				}
			}
			else {
				String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
				Path p = Paths.get(full_path);
				Files.createDirectories(p.getParent());
				Files.write(p, results.getBytes());
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}



	public static void main(String[] args) {

		String final_result = "";
		for(int i = 0; i < oa.length+1; i++) {
			networks[i] = factory.createClassificationNetwork(
					new int[] {inputLayer, hiddenLayer1,hiddenLayer2,  hiddenLayer3, outputLayer}, activation);
			if(i<3)
				nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
		}

		oa[0] = new RandomizedHillClimbing(nnop[0]);
		oa[1] = new SimulatedAnnealing(1e10, .35, nnop[1]);
		oa[2] = new StandardGeneticAlgorithm(100, 10, 30, nnop[2]);
		//OptimizationAlgorithm bp = new BatchBackPropagationTrainer(set, networks[i], measure, rule);
		//bckprp = new BatchBackPropagationTrainer(set, networks[3], measure, rule);


		int[] iterations = {10, 100, 500, 1000, 1500, 2000, 2500, 3000};
		//int[] iterations = {10, 20};
		
		String fileName = "diabetes_results.csv";
		for (int trainingIterations : iterations) {
			results = "";
			for (int i = 0; i < oa.length+1; i++) { //oa.length + 1 for extra backprop
				double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
				if(i == 3)
					train(new BatchBackPropagationTrainer(set, networks[i], new SumOfSquaresError(), rule), networks[i], oaNames[i], trainingIterations);
				else {
					train(oa[i], networks[i], oaNames[i], trainingIterations); //trainer.train();
					Instance optimalInstance = oa[i].getOptimal();
					networks[i].setWeights(optimalInstance.getData());
				}

				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);

				// Calculate Training Set Statistics //
				double predicted, actual;
				start = System.nanoTime();
				for (int j = 0; j < train_set.length; j++) {
					networks[i].setInputValues(train_set[j].getData());
					networks[i].run();

					actual = Double.parseDouble(train_set[j].getLabel().toString());
					predicted = Double.parseDouble(networks[i].getOutputValues().toString());

					// System.out.println("actual is " + actual);
					// System.out.println("predicted is " + predicted);

					double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

				}
				end = System.nanoTime();
				testingTime = end - start;
				testingTime /= Math.pow(10, 9);

				results += "\nTrain Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
						"\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
						+ df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
						+ " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

				final_result = oaNames[i] + "," + trainingIterations + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
				+ "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
				"," + df.format(testingTime);
				
				write_output_to_file("Results", fileName, final_result, true);

				// Calculate Test Set Statistics //
				start = System.nanoTime();
				correct = 0;
				incorrect = 0;
				for (int j = 0; j < test_set.length; j++) {
					networks[i].setInputValues(test_set[j].getData());
					networks[i].run();

					actual = Double.parseDouble(test_set[j].getLabel().toString());
					predicted = Double.parseDouble(networks[i].getOutputValues().toString());

					double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
				}
				end = System.nanoTime();
				testingTime = end - start;
				testingTime /= Math.pow(10, 9);

				results += "\nTest Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
						"\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
						+ df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
						+ " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

				final_result = oaNames[i] + "," + trainingIterations + "," + "testing accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
				+ "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
				"," + df.format(testingTime);
				
				write_output_to_file("Results", fileName, final_result, true);
			}
			System.out.println("results for iteration: " + trainingIterations + "\n---------------------------");
			System.out.println(results);
		}
	}

	private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
		//System.out.println("\nError results for " + oaName + "\n---------------------------");
		int trainingIterations = iteration;
		for(int i = 0; i < trainingIterations; i++) {
			oa.train();

			double train_error = 0;
			for(int j = 0; j < train_set.length; j++) {
				network.setInputValues(train_set[j].getData());
				network.run();

				Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
				example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
				train_error += measure.value(output, example);
			}


			//System.out.println("training error :" + df.format(train_error)+", testing error: "+df.format(test_error));
		}
	}
	private static void train(BatchBackPropagationTrainer oa, BackPropagationNetwork network, String oaName, int iteration) {
		//System.out.println("\nError results for " + oaName + "\n---------------------------");
		int trainingIterations = iteration;
		for(int i = 0; i < trainingIterations; i++) {
			oa.train();

			double train_error = 0;
			for(int j = 0; j < train_set.length; j++) {
				network.setInputValues(train_set[j].getData());
				network.run();

				Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
				example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
				train_error += measure.value(output, example);
			}


			//System.out.println("training error :" + df.format(train_error)+", testing error: "+df.format(test_error));

			//System.out.println("training error :" + df.format(train_error));
		}
	}



	private static Instance[] initializeInstances() {

		double[][][] attributes = new double[768][][];//max length of the data

		try {
			BufferedReader br = new BufferedReader(new FileReader(new File("D:/OMSCS/Github/CS7641-Machine Learning/CS7641-Assignment2/ABAGAIL/src/opt/test/pima_diabetes_preprocessed.csv")));//change the path correctly later

			//for each sample
			for(int i = 0; i < attributes.length; i++) {
				Scanner scan = new Scanner(br.readLine());
				scan.useDelimiter(",");

				attributes[i] = new double[2][];
				attributes[i][0] = new double[8]; // 8 attributes
				attributes[i][1] = new double[1]; // classification

				// read features
				for(int j = 0; j < 8; j++)
					attributes[i][0][j] = Double.parseDouble(scan.next());

				attributes[i][1][0] = Double.parseDouble(scan.next());
				//System.out.println(attributes[i][1][0]);

			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}

		Instance[] instances = new Instance[attributes.length];

		for(int i = 0; i < instances.length; i++) {
			instances[i] = new Instance(attributes[i][0]);
			instances[i].setLabel(new Instance(attributes[i][1][0]));
		}

		return instances;
	}
}
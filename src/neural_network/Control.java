package neural_network;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class Control {
	private int state;
	private BinaryClassifier binary;
	private MultiClassifier multi;
	
	public Control() {
		this.state = 0; 
	}

	
	
	
	/**
	 * Reading an entire NN from a file including weight anad bias values. 
	 * 
	 * @param input
	 * @return
	 */
	public String loadNN(String[] input) {
		File f = new File(input[1]);
		//Classifier params : array of ints [3,4,5] values are numbr of nodes for each layer
		//input should look something like: "create 2 3 2 3 or create 2,3,4,2
		return "";
	}
	

	
	public String changeparams(String input) {
		
		return "";
	}


	
	
	private String testNN(String input) {
		
		return "";
	}
	
	

	


	
	
	
	public static void main(String[] args) throws FileNotFoundException {
		
		Map<double[], double[]> instances = DataParser.readSupervisedInstances("C:\\Users\\Ashton\\eclipse-workspace\\NNImageClassifier\\src\\data\\iris.data.txt", 4, 3);
		
		int[] layout = new int[]{4,3,3};
		MultiClassifier classifier = new MultiClassifier(layout, 0.2, 0,0,0);
		List<Map<double[], double[]>> trainTestSplit = DataParser.trainTestSplit(instances, 0.8);
		Map<double[], double[]> train = trainTestSplit.get(0);
		Map<double[], double[]> test = trainTestSplit.get(1);
		for(int epoch = 0; epoch < 3; epoch ++) {
		    for(double[] instance : train.keySet()) {
	            classifier.train(instance, instances.get(instance), classifier.network);
	        }
		}
		
		int correct = 0;
		for(double[] instance : test.keySet()) {
		   double[] result = classifier.classify(instance, classifier.network);
		   
		   int prediction = 0;
		   double max = result[0];
		   for(int i = 1; i < result.length; i++) {
		       if(max < result[i]) prediction = i; max = result[i];
		   }
		   int actual = 0;
		   double[] target = test.get(instance); 
		   for(int i =0; i<target.length; i++) {
		       if(target[i] == 1) {
		           actual = i;
		           break;
		       }
		   }
		   if(prediction == actual) correct++;
		}	
		
		double accuracy = (double)correct / test.size();
		
		System.out.println("Classifier accuracy : "+accuracy);
		System.out.println("Correctly Indentified : "+correct);
		System.out.println("Incorrectly Identified: " +(test.size() - correct));
	}
	
	
	

}

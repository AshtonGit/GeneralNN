package neural_network;

import java.io.FileNotFoundException;

public class Main {
	
	public Main() {
		
	}
/**
 * 
 * NN general image classifier
	provide the images and this classifier will do the rest.
	interact through a swing UI
	classifiers stored in txt or .dat files
	classifiers architecture and weights are represented by a custom grammar
	program parses files written in the grammar to create the classifiers or trains its own.
	provided images are read as inputs into the classifier input nodes. 

	Grammar needs to convey two scenarios:	
		a. classifier with random weights, just the node architecture needed 
		b. pre built classifier with predefined weights, architecture and weights
		
 Steps, assume each step is tested and working before moving on to next one.
	1. NN with arguments
	2. grammar and grammar parser that construct the NN by passing it args
	3. 

 * @param args
 * @throws FileNotFoundException 
 */
	public static void main(String[] args) throws FileNotFoundException {
		DataParser dt = new DataParser("C:\\Users\\Ashton\\eclipse-workspace\\NNImageClassifier\\src\\data\\debug_parser.txt", 3);
		
		int[] architecture = new int[]{2,3,1};
		double[] input = new double[] {0.5, 0.99};
		Classifier NN = new Classifier(architecture);
		double[] output = NN.classify(input);
		System.out.println(output.toString());
	}
	
	
	public Classifier generateClassifierRandWeight() {
		
		return null;
	}

}

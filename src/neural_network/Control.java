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
		
		
	}
	
	
	

}

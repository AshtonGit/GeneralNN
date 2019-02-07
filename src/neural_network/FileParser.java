package neural_network;

import java.io.*;
import java.util.*;

public class FileParser {
	public FileParser() {
		
	}
	
	
	/**
	 * 
	 * CHANGE THIS WHEN YOU CHANGE THE CODE
	 * Why this class is begin built.
	 * 	Neural Networks need to be passed to the program through files.
	 * 	This class turns those files into an array of integers.
	 * 	In the future, i will implement the functionality of passing networks that have
	 * 	already been trained with the weights and biases set to specific decimal values
	 * Another functionality would be giving users ability to customize when
	 * creating new untrained networks.
	 * 
	 * Currently: 
	 *  Control uses this class to read the file and create an int array
	 *  which is passed to Classifier constructor to build that classifier. 
	 * @param file
	 * @return
	 */
	public int[] parseNetworkFile(String filename) {
		File file = new File(filename);
		Scanner sc;
		
		return new int[1];
	}
}

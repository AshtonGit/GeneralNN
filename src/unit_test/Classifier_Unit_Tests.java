package unit_test;
import org.junit.*;
import java.util.*;
import neural_network.Classifier;
public class Classifier_Unit_Tests {
	
	/**
	 * need a classifier with known starting values. 
	 * should have a few different input instances
	 * should know what the correct output and weight changes for those instances are
	 * test against those known inputs and outputs
	 * 
	 * 
	 * 
	 * new classifier constructer and constructer process
	 * @return
	 */
	
	double[][] test_instances = {{0.0, 0.0,},{1.0, 1.0},{-1.5, 0.5},{0.3, 0.5}, {30.0, 100.0}};
	
	public Classifier prebuiltClassifier(int select) {
		Classifier testClassifier;
				
		switch (select) {
		case 0:testClassifier = new Classifier(); //1 input, 1 output, 2 hidden nodes. super simple
			break;
		case 1: testClassifier = new Classifier();// 2 inputs, 3 hidden nodes, 2 outputs, more complex
			break;
		}
			
		
		return testClassifier;
	}
	
	public void testBackprop() {
		Classifier tc = new Classifier();
	}
	
	public void testWeightUpdate() {
		
	}
	
	public void testTraining() {
		
	}
	
	public void testExceptionHandling() {
		
	}
}

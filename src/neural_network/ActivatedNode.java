package neural_network;

import java.util.*;
public class ActivatedNode extends Node{

	

	private double bias;
	private double error_signal;
	private Map<Node, Double> weights;
	
	
	public ActivatedNode(double bias, double error_signal, Map<Node, Double> weights, ActivatedNode[] children) {
		super( 0.0, children);
		this.bias = bias;
		this.error_signal = error_signal;
		this.weights = weights;
		
	}
	
	/**
	 * node with the bias and weights randomly initialized from a given range.
	 */
	public ActivatedNode(double maxBias, double minBias, double minW, double maxW, double error_signal, Node[] parents, ActivatedNode[] children){
		super(0.0, children);
		Random rand = new Random();
		this.bias = minBias + (maxBias - minBias) * rand.nextDouble();	
		this.error_signal = 0;
		this.weights = new HashMap<Node, Double>();
		setParentRandWeight(parents, minW, maxW);		
		
	}
	
	public void setParentRandWeight(Node[] parents, double min, double max) {		
		Random rand = new Random();
		for(Node parent : parents) {			
			double weight = min + (max - min) * rand.nextDouble();
			weights.put(parent, weight);
		}
	}
	
	public Map<Node, Double> getWeights(){
		return this.weights;
	}
	public void setWeights(Map<Node, Double> weights) {
		this.weights = weights;
	}	
	
	public Set<Node> getParentNodes() {
		return this.weights.keySet();
	}
	
	public double getBias() {
		return this.bias;
	}
	
	public void setBias(double bias) {
		this.bias = bias;
	}
	
	public void setErrorSignal(double error_signal) {
	    this.error_signal = error_signal;
	}
	
	public double getErrorSignal() {
	    return this.error_signal;
	}
	
	/**
	 *classifier will select the activation function. 
	 *this allows user to dynamically select what activation function they want for a particular classifier
	 * @return
	 */
	public double evaluate() {
		double sum = 0.0;
		Set<Node> iter = weights.keySet();
		for(Node parent : iter) {
			sum += (parent.getOutput() * weights.get(parent));
		}
		sum += this.bias;
		return sum;
	}

	
	
}

package neural_network;

import java.util.*;
public class ActivatedNode extends Node{

	
	/**
	 * create a node interface?? hidden node, input node, output node all fufill it
	 * 
	 * how should bias be handled when coming to creating new nodes
	 */
	private double bias;
	private double output;
	private double error_signal;
	private Map<Node, Double> weights;
	private ActivatedNode[] children;
	
	public ActivatedNode(double bias, Map<Node, Double> weights, ActivatedNode[] children) {
		super(0.0);
		this.bias = bias;
		this.weights = weights;
		this.children = children;
	}
	
	/**
	 * node with random params and no parents yet
	 */
	public ActivatedNode(double max, double min) {
		super(0.0);
		Random rand = new Random();
		this.bias = min + (max - min) * rand.nextDouble();
		this.weights = new HashMap<Node, Double>();
	}
	
	public ActivatedNode(double bias) {
		super(0.0);
		this.bias = bias;
		this.weights= new HashMap<Node, Double>();
	}
	
	public void setParentRandWeight(Node[] parent_array	, double min, double max) {
		//randomly generate weights
		int len = parent_array.length;
		Random rand = new Random();
		for(int i=0; i< len; i++) {			
			double weight = min + (max - min) * rand.nextDouble();
			weights.put(parent_array[i], weight);
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
	
	public ActivatedNode[] getChildren() {
		return this.children;
	}
	
	public void setChildren(ActivatedNode[] children) {
		this.children = children;
	}
	
	public void setErrorSignal(double error_signal) {
		this.error_signal = error_signal;
	}
	
	public double getErrorSignal() {
		return this.error_signal;
	}
	
	public double getBias() {
		return this.bias;
	}
	
	public void setBias(double bias) {
		this.bias = bias;
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
	
	public void setOutput(double output) {
		this.output = output;
	}
	
	public double getOutput() {		
		return this.output;
	}
	
	
}

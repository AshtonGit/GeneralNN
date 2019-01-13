package neural_network;

import java.util.*;
public class Classifier {

	/**
	 * to do (in no particular order)
	 * 	in this class, many times have to cast Node[] as (ActivatedNode[])Node. It would prob be prefereable to do this the other way around
	 *  where List<ActivatedNode[] network; and cast ActivatedNode[] as (Node[])ActivatedNode.  
	 * 	
	 * 
	 * 
	 * create some kind of error handling stack / procedure / standard when throwing errors
	 */
	
	
	/**
	 * consider how to store the data structure
	 * alternatives: 
	 * 	a. list of arrays of nodes, layer number is implied by position in the list
	 * 	b. map of list of nodes with their layer number
	 * 	c. list of the initial and output nodes
	 * 
	 */
	
	private List<Node[]> network;
	private double n_learn;
	private double dmax;
	private double momentum;
	private double flat_elim;
	//epochs? training freq?
	

	public Classifier(int[] architecture) {		
		this.network = new ArrayList<Node[]>();
		int size = architecture.length;				
		for(int i = 0; i< size - 1 ; i++) {			
			int numNodes = architecture[i];
			Node[] layer = createLayer(numNodes, i);
			this.network.add(layer);
		}
		//add the final layer, outisde of loop for -1 layerPos arg
		this.network.add(createLayer(architecture[size-1], -1));
		
		connectLayers();
	}
	
	/**
	 * if layerpos = 0, no parents
	 * if layerpos = final layer, no children
	 * @param numNodes
	 * @param layerPos
	 * @return
	 */
	private ActivatedNode[] createLayer(int numNodes, int layerPos) {
		ActivatedNode[] layer = new ActivatedNode[numNodes];
		
		for(int i=0; i < numNodes; i++) {
			layer[i] = new ActivatedNode(-1.0, 1.0); //maxBias = 1, minBias = -1
		}
		
	
		return layer;
	}
		
	private void connectLayers() {
		
		int len = this.network.size();
		//only do hidden and input layer, output doesnt need to be run as no children to connect with
		for(int i=0; i < len - 2; i++) {
			Node[] parents = this.network.get(i);
			ActivatedNode[] children = (this.network.get(i))[0].getChildren();
			int p_len = parents.length;
			int c_len = parents.length;
			
			for(int p = 0; p < p_len; p++) {
				parents[p].setChildren(children);
			}
			
			for(int c = 0 ; c< c_len; c++) {
				children[c].setParentRandWeight(parents, -1.0, 1.0);
			}
		}
	}

	
	public void train(List<double[]> trainingset, double[] targets) {				
		
		for(double[] instance: trainingset) {
			classify((instance)); 	
			backPropogate(targets);
			
			int num_layer = network.size(); //update weights and biases
			
			for(int i=1; i<num_layer; i++) {//start from i=1 to skip the input layer
				// need to incorporate momentum and flat_elim, need to use inputs, so use the array method
				
				ActivatedNode[] layer = (ActivatedNode[]) this.network.get(i); 
				int layer_len = layer.length;
				for(int j = 0; j < layer_len; j++) {
					ActivatedNode neuron = layer[j];
					Map<Node, Double> weights = neuron.getWeights();
					Set<Node> parents = neuron.getParentNodes();
					for(Node parent: parents) {
						double old_weight = weights.get(parent);
						double	weight_delta = n_learn * neuron.getErrorSignal() * parent.getOutput();
						weights.replace(parent, old_weight + weight_delta);
					}
					double new_bias = neuron.getBias() + (n_learn * neuron.getErrorSignal());
					neuron.setBias( new_bias );
				}
				
			}				
		}
	}
		
		
		
	
	
	public double test(List<Double[]> testset) {
		//return accuracy?? or an lits of results
		return 0.0;
	}
	
	public double[] classify(double[] input) {
		//give all input nodes the inputs
		//forward propogate
		//get results,
		//back propogate??
		applyInput(input);
		return forwardPropogate();		
	}
	/**
	 * sets values of the input nodes in networks first layer.
	 * values are doubles stored in an array
	 * 
	 *  TO DO:
	 *  **ensure inputs match number of outputs else throw error
	 * @param input
	 */
	public void applyInput(double[] input) {
		Node[] layer = network.get(0);
		int len = layer.length;
		if(input.length < len)System.out.println("Error, insufficient number of inputs"); //make this throw an exception
		else {
			for(int i=0; i<len; i++) {
				layer[i].setOutput(input[i]);
			}
		}
	}
	
	/**
	 * do input nodes need their values to be weighed against the bias and or put through
	 * the activation function??
	 */
	public double[] forwardPropogate() {
		int layer_count = network.size();		
		
		for(int i = 1; i < layer_count; i++) {
			ActivatedNode[] layer = (ActivatedNode[])network.get(i);
			int node_count = layer.length;
			for(int j=0; j<node_count; j++) {
				double output = layer[j].evaluate(); //node takes care of evaluation itself
				output = layer[j].activationSigmoid(output);
				layer[j].setOutput(output);
			}
		}
		
		//return output layer values as results
		ActivatedNode[] output_layer = (ActivatedNode[])network.get(layer_count-1);
		int len = output_layer.length;
		double[] results = new double[len];		
		for(int i = 0; i < len; i++) {
			results[i] = output_layer[i].getOutput();
		}		
		return results;
	}
	
	/**
	 * Error is squared so that smaller errors have a smaller effect on the weight updates, while
	 * larger errors are emphasized even moreso in their effect on the new weight values. 
	 * calculating error using the simpler method found in  https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
	 *Stochastic gradient descent. 
	 * @param result
	 * @param target
	 */
	public void backPropogate(double[] target) {
		// Does this method correctly handle input nodes and their lack of effect on weights??
		int output_indx = network.size() - 1;
		for(int i = output_indx; i >= 0; i--) { //move through network in reverse order, starting at output layer.
			Node[] layer = network.get(i);
			List<Double> errors = new ArrayList<Double>();
			int num_nodes = layer.length;
			
			if(i == output_indx) {
				for(int j =0; j<num_nodes; j++) {
					Node neuron = layer[j];
					double neuron_error = Math.pow(target[i] - neuron.getOutput(), 2) / 2;
					errors.add(neuron_error);
				}
			}else { //hidden layer
				for(int j =0; j<num_nodes; j++) {
					Node neuron = layer[j];
					double neuron_error = 0.0;
					for(ActivatedNode child: neuron.getChildren()) {
						double weight = child.getWeights().get(neuron);
						neuron_error += weight * child.getErrorSignal();
					}
					errors.add(neuron_error);
				}
			}
			//update neuron error signals
			for(int j = 0; j < num_nodes; j++) {
				Node neuron = layer[j];
				neuron.setErrorSignal(errors.get(j) * transferDerivative(neuron.getOutput()));
			}
			
					
		}				
	}
	
	/**
	 * Used to calculate the slope of an output neuron for neurons that used the sigmoid 
	 * function as their activator.
	 *
	 * @param output
	 * @return
	 */
	private double transferDerivative(double output) {
		return output*(1 - output);
	}
	
	private double outputNodeError(double output, double target) {
		double error = Math.pow(target - output, 2)/2;
		return error * transferDerivative(output);
	}
	
	/**
	 * jth neuron is current neuron, kth neuron is neuron in prev / parent layer.
	 * @param weight_k weight that connects kth neuron to jth neuron
	 * @param error_j error signal from jth neuron 
	 * @param output output from current neuron
	 * @return
	 */	
	private double hiddenNodeError(double weight_k, double error_j, double output) {
		double error = (weight_k * error_j) * transferDerivative(output);
		return error;
	}
	
}

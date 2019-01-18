package neural_network;

import java.util.*;
public class Classifier {

	/**
	 * to do (in no particular order)
	 * 	in this class, many times have to cast Node[] as (ActivatedNode[])Node. It would prob be prefereable to do this the other way around
	 *  where List<ActivatedNode[] architecture; and cast ActivatedNode[] as (Node[])ActivatedNode.  
	 * 	
	 * Issue with weight_delta:
	 * 	some examples have weight -= weight * delta
	 * other examples have : weight += weight * delta
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
	private Network architecture;
	
	

	public Classifier(int[] architecture) {		
		List<Node[]> node_layers = new ArrayList<Node[]>();
		int size = architecture.length;				
		for(int i = 0; i< size - 1 ; i++) {			
			int numNodes = architecture[i];
			Node[] layer = createLayer(numNodes, i);
			node_layers.add(layer);
		}
		//add the final layer, outisde of loop for -1 layerPos arg
		node_layers.add(createLayer(architecture[size-1], -1));
		this.architecture = new Network(node_layers, 0,0,0,0);
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
		List<Node[]> layers = this.architecture.getNodeNetwork();
		int len = layers.size();
		//only do hidden and input layer, output doesnt need to be run as no children to connect with
		for(int i=0; i < len - 2; i++) {
			Node[] parents = layers.get(i);
			ActivatedNode[] children = layers.get(i)[0].getChildren();
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

	
	public void train(double[] instance, double[] targets) {		
		classify(instance); 	
		backPropogate(targets);		
		
		List<Node[]> layers = architecture.getNodeNetwork();
		double n_learn = architecture.getLearnRate();
		int num_layer = layers.size(); //update weights and biases
		for(int i=1; i<num_layer; i++) {//start from i=1 to skip the input layer
		// need to incorporate momentum and flat_elim, need to use inputs, so use the array method
			ActivatedNode[] layer = (ActivatedNode[]) layers.get(i); 
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
		List<Node[]> layers = architecture.getNodeNetwork();
		int output_indx = layers.size() - 1;
		for(int i = output_indx; i >= 0; i--) { //move through architecture in reverse order, starting at output layer.
			Node[] layer = layers.get(i);
			List<Double> errors = new ArrayList<Double>();
			int num_nodes = layer.length;			
			if(i == output_indx) { //output layer
				for(int j =0; j<num_nodes; j++) {					
					double neuron_error = target[i] - layer[j].getOutput();
					errors.add(neuron_error);
				}
			}else { //hidden layer
				for(int j =0; j<num_nodes; j++) {
					Node neuron = layer[j];
					double neuron_error = 0.0;
					for(ActivatedNode child: neuron.getChildren()) {
						double weight = child.getWeights().get(neuron);
						neuron_error += (weight * child.getErrorSignal());
					}
					errors.add(neuron_error);
				}
			}
			//update neuron error signals
			for(int j = 0; j < num_nodes; j++) {
				Node neuron = layer[j];
				neuron.setErrorSignal(errors.get(j) * transferDerivativeSigmoid(neuron.getOutput()));
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
	 * sets values of the input nodes in architectures first layer.
	 * values are doubles stored in an array
	 * 
	 *  TO DO:
	 *  **ensure inputs match number of outputs else throw error
	 * @param input
	 */
	public void applyInput(double[] input) {
		Node[] layer = architecture.getNodeNetwork().get(0);
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
		List<Node[]> layers = architecture.getNodeNetwork();
		int layer_count = layers.size();		
		for(int i = 1; i < layer_count; i++) {
			ActivatedNode[] layer = (ActivatedNode[])layers.get(i);
			int node_count = layer.length;
			for(int j=0; j<node_count; j++) {
				double output = layer[j].evaluate(); //node takes care of evaluation itself
				output = activationSigmoid(output);
				layer[j].setOutput(output);
			}
		}
		//return output layer values as results
		ActivatedNode[] output_layer = (ActivatedNode[])layers.get(layer_count-1);
		int len = output_layer.length;
		double[] results = new double[len];		
		for(int i = 0; i < len; i++) {
			results[i] = output_layer[i].getOutput();
		}		
		return results;
	}
	
	
	
	/**
	 * Used to calculate the slope of an output neuron for neurons that used the sigmoid 
	 * function as their activator.
	 *
	 * @param output
	 * @return
	 */
	private double transferDerivativeSigmoid(double output) {
		return output*(1 - output);
	}
	
	
	private double meanSquaredError(double target, double output) {
		return Math.pow(target - output, 2) / 2;
	}
	
	/**
	 * Activation function
	 * @param output
	 * @return
	 */
	private double activationSigmoid(double output) {
		double euler = Math.exp(-output);
		return 1 / (1 + euler );
	}
	
	
	public Network getArchitecture(){
		return this.architecture;
	}
	
	
	
}

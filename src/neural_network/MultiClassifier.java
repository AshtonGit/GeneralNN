package neural_network;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Used to make class predictions where there are multiple possible classes
 * where (Number output nodes > 1)
 * @param input
 * @return
 */
public class MultiClassifier extends Classifier{

	private Network architecture;
	public MultiClassifier(int[] architecture) {
		super(architecture);
		this.architecture = super.getArchitecture();
		// TODO Auto-generated constructor stub
	}

	
	
	@Override
	public double[] forwardPropogate() {
		return null;
	}
	
	@Override
	/**
	 *If NN fails to learn correctly, the issue may be that weight delta is added to old weight instead of subtracted. 
	 *	Have found both methods used in different sources / texts. My interpretation may be incorrect 
	 */
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

	
	
	@Override
	public void backPropogate(double[] target) {
		List<Node[]> layers = this.architecture.getNodeNetwork();
		int output_indx = layers.size() - 1;
		for(int i = output_indx; i >= 0; i--) { //move through network in reverse order, starting at output layer.
			Node[] layer = layers.get(i);
			int num_nodes = layer.length;			
			if(i == output_indx) { //output layer				
				for(int j =0; j<num_nodes; j++) {					
					layer[j].setErrorSignal(layer[j].getOutput() - target[j]);
				}
			}else { //hidden layer
				for(int j =0; j<num_nodes; j++) {
					Node neuron = layer[j];
					double neuron_error = 0.0;
					for(ActivatedNode child: neuron.getChildren()) {
						double weight = child.getWeights().get(neuron);
						neuron_error += (weight * child.getErrorSignal());
					}					
					neuron.setErrorSignal(neuron_error * transferDerivativeSigmoid(neuron.getOutput()));
				}				
			}			
		}
	}
	
	
	
	
	
	private double transferDerivativeSigmoid(double output) {
		return output*(1 - output);
	}
	
	
	private double crossEntropyCost(double[] target, double[] results) {
		double error = 0.0;
		double len = target.length;
		for(int i=0; i<len; i++) {
			error -= (results[i] * Math.log(target[i]));
		}
		return error;
	}
	
	
	private double[] activationSoftMax(double[] output) {
		double[] softmax = new double[output.length];
		double sum = 0.0;
		double len = output.length;
		double e = Math.E;
		for(double o: output) {
			sum += Math.pow(e, o);
		}
		for(int i = 0; i<len; i++) {
			softmax[i] = Math.pow(e, output[i]) / sum;
		}
		return softmax;
	}
}

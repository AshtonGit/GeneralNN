package neural_network;

import java.util.*;
public class BinaryClassifier extends Classifier{


	

	
/**
 * 
 * 
 * @param layout
 */
	public BinaryClassifier() {	
	    super();	
	}
	
	
	
	
	/**
	 * 
	 *binary classification
	 * @param result
	 * @param target
	 */
	public List<Node[]> backPropagate(double[] target, List<Node[]> network ) {
		int output_indx = network.size() - 1;
		//move through network backwards, starting at output layer. Ignore input nodes as they dont require error signal
		for(int i = output_indx; i >= 0; i--) { 
			ActivatedNode[] layer = (ActivatedNode[]) network.get(i);	
			double neuron_error = 0.0;
			if(i == output_indx) {									
				ActivatedNode neuron = layer[0];
			    neuron_error = target[0] - neuron.getOutput();				
				neuron.setErrorSignal(neuron_error * transferDerivativeSigmoid(neuron.getOutput()));			
			}else {
				for(ActivatedNode neuron  : layer) {					
					neuron_error = 0.0;
					for(ActivatedNode child: neuron.getChildren()) {
						double weight = child.getWeights().get(neuron);
						neuron_error += (weight * child.getErrorSignal());
					}
					neuron.setErrorSignal(neuron_error * transferDerivativeSigmoid(neuron.getOutput()));       
				}
			}			
		}		
		return network;
	}

	
	
	
		
	/**
	 *  TO DO:
	 *  **ensure inputs match number of outputs else throw error
	 * @param input
	 */
	
    @Override
    public double[] classify(double[] input, List<Node[]> network) {
        Node[] input_layer = network.get(0);
        int len = input_layer.length;
        if(input.length < len)System.out.println("Error, insufficient number of inputs"); //make this throw an exception
        else {
            for(int i=0; i<len; i++) {
                input_layer[i].setOutput(input[i]);
            }
        }
        int layer_count = network.size();      
        for(int i = 1; i < layer_count; i++) {
            ActivatedNode[] layer = (ActivatedNode[])network.get(i);
            int node_count = layer.length;
            for(int j=0; j<node_count; j++) {
                double output = layer[j].evaluate(); //node takes care of evaluation itself
                output = activationSigmoid(output);
                layer[j].setOutput(output);
            }
        }
        //return output layer values as results
        ActivatedNode[] output_layer = (ActivatedNode[])network.get(layer_count-1);
        len = output_layer.length;
        double[] results = new double[len];     
        for(int i = 0; i < len; i++) {
            results[i] = output_layer[i].getOutput();
        }       
        return results;
    }
	
	
	
	
	
	
	
	
	
}

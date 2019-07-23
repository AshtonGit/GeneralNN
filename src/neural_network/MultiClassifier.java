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

	
	public MultiClassifier() {
		super();
	}

   	@Override
	public List<Node[]> backPropagate(double[] target, List<Node[]> network) {
		int output_indx = network.size() - 1;
		for(int i = output_indx; i > 0; i--) { //move through network in reverse order, starting at output layer.
			ActivatedNode[] layer = (ActivatedNode[])network.get(i);
			int num_nodes = layer.length;			
			if(i == output_indx) { //output layer				
				for(int j =0; j<num_nodes; j++) {
				    double error_signal = layer[j].getOutput() - target[j];				    
					layer[j].setErrorSignal(error_signal);
				}
			}else {
				for(ActivatedNode neuron : layer) {					
					double neuron_error = 0.0;
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
     * sets values of the input nodes in networks first layer.
     * values are doubles stored in an array
     * 
     *  TO DO:
     *  **ensure inputs match number of outputs else throw error
     * @param input
     */
    @Override
    public double[] classify(double[] input, List<Node[]> network) {
        //apply inputs
        validateInputData(input, network);
        Node[] input_layer = network.get(0);
        int len = input_layer.length;
        for(int i=0; i<len; i++) {
            input_layer[i].setOutput(input[i]);
        }
        //classify instances
        int layer_count = network.size();       
        for(int i = 1; i < layer_count; i++) {
            ActivatedNode[] layer = (ActivatedNode[])network.get(i);
            int node_count = layer.length;
            if(i == layer_count -1) { //final layer of network uses softmax activation funciton
                double[] pre_activ = new double[node_count];
                for(int j=0; j<node_count; j++) {
                    pre_activ[j] = layer[j].evaluate();                 
                }
                double[] activated = activationSoftMax(pre_activ);
                for(int j=0; j<node_count; j++) {
                    layer[j].setOutput(activated[j]);
                }
            }else {
                for(int j=0; j<node_count; j++) {
                    double output = layer[j].evaluate();
                    output = super.activationSigmoid(output);
                    layer[j].setOutput(output);
                }
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
    
    
    public static double[] activationSoftMax(double[] output) {
        double[] softmax = new double[output.length];
        double sum = 0.0;
        int len = output.length;
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

package neural_network;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class Classifier {
    
    

    List<Node[]> network;
    double n_learn;
    double dmax;
    double momentum;
    double flat_elim;
    /**
     * issues
     * 1. Input node initialization assumes that number of layers > 1. Need asserts that 
     * ensure that number of layers >= 3 so there is input layer, hidden layer, output layer
     *  
     * 
     * @param layout
     * @param n_learn
     * @param dmax
     * @param momentum
     * @param flat_elim
     */
    
    public Classifier(int[] layout, double n_learn, double dmax, double momentum, double flat_elim) {   
        this.n_learn = n_learn;
        this.dmax = dmax;
        this.momentum = momentum;
        this.flat_elim = flat_elim;
        this.network = new ArrayList<Node[]>();        
        int size = layout.length;           
        int num_nodes = layout[0];
        Node[] input_layer = new Node[num_nodes];
        for(int i=0; i<num_nodes; i++) {            
            input_layer[i] = new Node(0.0, layout[1]);            
        }
        network.add(input_layer);
        for(int i = 1; i< size; i++) {          
            num_nodes = layout[i];
            ActivatedNode[] layer = new ActivatedNode[num_nodes];
            for(int j =0; j< num_nodes; j++) {
               layer[j] = new ActivatedNode(-1.0, 1.0, -1.0, 1.0, 0.0, network.get(i-1), null); //initial maxBias = 1 and minBias = -1
            }           
            network.add(layer); 
        }
        network = connectLayers(network);
        
    }
    
    public abstract double[] classify(double[] input, List<Node[]> network);
    
    public abstract List<Node[]> backPropagate(double[] target, List<Node[]> network );
    
    /**
     * 
     * @param layers
     * @return
     */
    private List<Node[]> connectLayers(List<Node[]> network) {        
        int len = network.size();
        //only do hidden and input layer, output doesnt need to be run as no children to connect with
        for(int i=0; i < len - 1; i++) {
            Node[] parents = network.get(i);
            ActivatedNode[] children = (ActivatedNode[])network.get(i+1);
            int p_len = parents.length;
            int c_len = children.length;
            
            for(int p = 0; p < p_len; p++) {
                parents[p].setChildren(children);
            }
            
            for(int c = 0 ; c< c_len; c++) {
                children[c].setParentRandWeight(parents, -1.0, 1.0);
            }
        }        
        return network;
    }
    
    
    /**
     *If NN fails to learn correctly, the issue may be that weight delta is added to old weight instead of subtracted. 
     *  Have found both methods used in different sources / texts.
     */
    public void train(double[] instance, double[] targets, List<Node[]> network) {        
        classify(instance, network);     
        network = backPropagate(targets, network);    
        int num_layer = network.size(); //update weights and biases
        
        for(int i=1; i<num_layer; i++) {//start from i=1 to skip the input layer
        
            ActivatedNode[] layer = (ActivatedNode[]) network.get(i); 
            for(ActivatedNode neuron : layer) {             
                Map<Node, Double> weights = neuron.getWeights();
                Set<Node> parents = neuron.getParentNodes();
                for(Node parent: parents) {
                    double old_weight = weights.get(parent);
                    double  weight_delta = n_learn * neuron.getErrorSignal() * parent.getOutput();
                    weights.replace(parent, old_weight - weight_delta);                                     
                }
                double new_bias = neuron.getBias() - (n_learn * neuron.getErrorSignal());
                neuron.setBias( new_bias );
            }
        }               
    }
    
    protected double meanSquaredError(double target, double output) {
        return Math.pow(target - output, 2) / 2;
    }
    

    
    /**
     * Activation function
     * @param output
     * @return
     */
    protected double activationSigmoid(double output) {
        double euler = Math.exp(-output);
        return 1 / (1 + euler );
    }
    

    /**
     * Used to calculate the slope of an output neuron for neurons that used the sigmoid 
     * function as their activator.
     *
     * @param output
     * @return
     */
    protected double transferDerivativeSigmoid(double output) {
        return output*(1 - output);
    }
        
    
    public List<Node[]> getNetwork(){
        return this.network;
    }
    
    public double getLearnRate() {
        return this.n_learn;
    }
    
    public double getMomentum() {
        return this.momentum;
    }
    
    
    
    
    
    
    
    
}

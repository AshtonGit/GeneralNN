package neural_network;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class Classifier {
    
    

    public Classifier(){
        
    }
    
    public abstract double[] classify(double[] input, List<Node[]> network) throws Exception;
    
    public abstract List<Node[]> backPropagate(double[] target, List<Node[]> network );
    
    public List<Node[]> buildNeuralNetwork(int[] layout, double minW, double maxW){
        validateNetworkLayout(layout);        
        List<Node[]> network = new ArrayList<Node[]>();
         int layerCount = layout.length;
         int nodeCount = layout[0];
         Node[] input_layer = new Node[nodeCount];
         for(int i=0; i<nodeCount; i++) {
             input_layer[i] = new Node(0.0, layout[1]);
         }
         network.add(input_layer);
         for(int i =1; i<layerCount; i++) {
             nodeCount = layout[i];
             ActivatedNode[] layer = new ActivatedNode[nodeCount];
             for(int j =0; j<nodeCount; j++) {
                 //cannot provide the children arg to constructor yet as that layer has not been created yet
                 layer[j] = new ActivatedNode(minW, maxW, 0.0, network.get(i-1), null);
             }
             network.add(layer);
         }
         //connect the parents to their children, skipping output layer as it doesnt have children to connect with
         int len = network.size();         
         for(int i=0; i < len - 1; i++) {             
             ActivatedNode[] children = (ActivatedNode[])network.get(i+1);        
             for(Node parent : network.get(i))parent.setChildren(children);                         
         }        
         return network;
    }
    
    
    /**
     *If NN fails to learn correctly, the issue may be that weight delta is added to old weight instead of subtracted. 
     *  Have found both methods used in different sources / texts.
     */
    public List<Node[]> train(double[] instance, double[] target, List<Node[]> network, double learn_rate) throws Exception{
        validateTargetData(target, network);
        validateLearningParams(learn_rate, 0.0,0.0,0.0);
        classify(instance, network);     
        network = backPropagate(target, network);    
        int num_layer = network.size(); //update weights and biases
        
        for(int i=1; i<num_layer; i++) {//start from i=1 to skip the input layer
        
            ActivatedNode[] layer = (ActivatedNode[]) network.get(i); 
            for(ActivatedNode neuron : layer) {             
                Map<Node, Double> weights = neuron.getWeights();
                Set<Node> parents = neuron.getParentNodes();
                for(Node parent: parents) {
                    double old_weight = weights.get(parent);
                    double  weight_delta = learn_rate * neuron.getErrorSignal() * parent.getOutput();
                    weights.replace(parent, old_weight - weight_delta);                                     
                }
                double new_bias = neuron.getBias() - (learn_rate * neuron.getErrorSignal());
                neuron.setBias( new_bias );
            }
        }           
        return network;
    }
    
    
    public static void validateLearningParams(double n_learn, double dmax, double momentum, double flat_elim) throws IllegalArgumentException{
        if(n_learn <= 0) throw new IllegalArgumentException("Learning rate must be greater than 0");
        if(dmax < 0) throw new IllegalArgumentException("dmax must be greater than or equal to 0");
        if(momentum < 0)throw new IllegalArgumentException("momentum must be greater than or equal to 0");
        if(flat_elim < 0)throw new IllegalArgumentException("flat_elim must be greater than or equal to 0");
       
    }
    
    public static void validateNetworkLayout(int[] layout) throws IllegalArgumentException{
        if(layout.length < 3)throw new IllegalArgumentException("Insufficient layout length. Networks require a minimum of 3 layers");        
        for(int x : layout)if(x < 1) throw new IllegalArgumentException("Each layer requires at least 1 Node");
    }
    
    public static void validateInputData(double[] instance, List<Node[]> network) {
        Node[] inputLayer = network.get(0);
        if(instance.length != inputLayer.length)throw new IllegalArgumentException("Number of data points for instance must match number of inputs for network");
    }
    
    public static void validateTargetData(double[] instance, List<Node[]> network) {
        Node[] outputLayer = network.get(network.size() - 1);
        if(instance.length != outputLayer.length) throw new IllegalArgumentException("Number of data points for instance must match number of outputs for network");
    }
    
    public static double meanSquaredError(double target, double output) {
        return Math.pow(target - output, 2) / 2;
    }
    

       
    /**
     * Activation function
     * @param output
     * @return
     */
    public static double activationSigmoid(double output) {
        double euler = Math.exp(-output);
        return 1 / (1 + euler );
    }
    
    /**
     * reLU activation function. 
     * @param output
     * @return
     */
    public static double activationRelu(double output) {
        if(output < 0)return output * 0.01;
        else {
            return output;
        }
    }

    /**
     * Used to calculate the slope of an output neuron for neurons that used the sigmoid 
     * function as their activator.
     *
     * @param output
     * @return
     */
    public static double transferDerivativeSigmoid(double output) {
        return output*(1 - output);
    }
        
    
   
 
}



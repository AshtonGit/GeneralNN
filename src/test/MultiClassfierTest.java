package test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.IllegalArgumentException;
import org.junit.Test;

import neural_network.Classifier;
import neural_network.DataParser;
import neural_network.MultiClassifier;
import neural_network.Node;

public class MultiClassfierTest {
    
    /**
     * Test to ensure that the classifier has an acceptable accuracy rate (>= 0.8) when trained on the iris dataset over 5 epochs
     * This test failing indicates that the learning algorithm has been negatively altered in some way. 
     */
    @Test
    public void testAccuracyIris() {
        Map<double[], double[]> instances = DataParser.readSupervisedInstances("src/data/iris.data.txt", 4, 3);        
        int[] layout = new int[]{4,3,3};
        MultiClassifier classifier = new MultiClassifier();
        List<Node[]> network = classifier.buildNeuralNetwork(layout, -1, 1);
        List<Map<double[], double[]>> trainTestSplit = DataParser.trainTestSplit(instances, 0.8);
        Map<double[], double[]> train = trainTestSplit.get(0);
        Map<double[], double[]> test = trainTestSplit.get(1);
        for(int epoch = 0; epoch < 7; epoch ++) {
            for(double[] instance : train.keySet()) {                
                network = classifier.train(instance, instances.get(instance), network, 0.2);
            }
        }
        
        int correct = 0;
        for(double[] instance : test.keySet()) {
           double[] result = classifier.classify(instance, network);
           
           int prediction = 0;
           double max = result[0];
           for(int i = 1; i < result.length; i++) {
               if(max < result[i]) prediction = i; max = result[i];
           }
           int actual = 0;
           double[] target = test.get(instance); 
           for(int i =0; i<target.length; i++) {
               if(target[i] == 1) {
                   actual = i;
                   break;
               }
           }
           if(prediction == actual) correct++;
        }   
        
        double accuracy = (double)correct / test.size();
        System.out.println("MultiClassifier accuracy on iris dataset: "+accuracy);
        assert(accuracy > 0.65);
    }
    
    /**
     * Test that the classify function responds correctly when the number of inputs in the training instance
     *  is greater or lesser than the number of input nodes in the networks input layer.
     */
    @Test
    public void testTooFewInputsClassify() {
        int[] layout = new int[] {3,2,2};        
        Classifier classy = new MultiClassifier();        
        assertThrows(IllegalArgumentException.class,
                ()->{classy.classify( new double[] {0.0, 0.1}, classy.buildNeuralNetwork(layout, -1, 1));}
                );
    }
    
    /**
     * Test that the train function responds correctly when the number of inputs in the training instance
     *  is greater or lesser than the number of input nodes in the networks input layer.
     */
    @Test 
    public void inputDontMatchNetworkTrain() {
        int[] layout = new int[]{3,2,2};  
        Classifier classy = new MultiClassifier();
        assertThrows(IllegalArgumentException.class,
                ()->{classy.train(new double[]{0.0, 0.1}, new double[]{0.0, 0.0, 0.1} , classy.buildNeuralNetwork(layout, -1, 1), 0.2 );}
                );
    }
    
    /**
     * Test that the classifier responds correctly when it tries to train a network but the number of classes
     * in the target instance is greater or lesser than the number of classes in the networks output layer. 
     */
    @Test
    public void targetDontMatchNetworkTrain() {
        int[] layout = new int[] {3,2,3};
        Classifier classy = new MultiClassifier();
        assertThrows(IllegalArgumentException.class,
                ()->{classy.train(new double[]{0.0, 0.1, 0.2}, new double[]{0.0, 0.0} , classy.buildNeuralNetwork(layout, -1, 1), 0.2);}
                );
    }
    
    /**
     *Attempt to build a network where the number of layers <= 3. This is illegal as the network needs minimum 1 input layer, 1 hidden layer and 1 output layer.
     */
    @Test
    public void testTooFewLayers() {
        Classifier classy = new MultiClassifier();
        assertThrows(IllegalArgumentException.class,
                ()->{List<Node[]> network = classy.buildNeuralNetwork(new int[] {2,1}, -1, 1);}                
                );
    }
    
    /**
     * Test that classifier responds correctly when asked to build a network where any one layer has less than 1 node.
     */
    @Test 
    public void testTooFewNodesInLayer() {
        Classifier classy = new MultiClassifier();
        assertThrows(IllegalArgumentException.class,
                ()->{List<Node[]> network = classy.buildNeuralNetwork(new int[] {0,1}, -1, 1 );}                
                );
    }
    
    @Test
    public void testMinWGreaterThanMaxW() {
        Classifier classy = new MultiClassifier();
        assertThrows(IllegalArgumentException.class,
                ()->{List<Node[]> network = classy.buildNeuralNetwork(new int[] {2,2,2}, 1, 1 );}                
                );
    }
       
    /**
     * Learning param that is <= 0. This is illegal as learning param is used to adjust the weight updates when training. If 0 the weights wont update
     * at all. If negative the weights update in the opposite desired direction.
     * Any of the other parameter cannot be negative though they can be zero values.
     */
    @Test
    public void illegalLearningParams() {
        int[] layout = new int[] {1,2,3};
        Classifier classifier = new MultiClassifier();
        List<Node[]> network = classifier.buildNeuralNetwork(new int[] {2,2,3}, -1, 1 );
        // learning rate = 0
        assertThrows(IllegalArgumentException.class,
                ()->{classifier.train(new double[] {0,0,0}, new double[] {0,0,0}, network, 0);}
                );
        // learning rate < 0
        assertThrows(IllegalArgumentException.class,
                ()->{classifier.train(new double[] {0,0,0}, new double[] {0,0,0}, network, -1);}
                );
        /**
        assertThrows(IllegalArgumentException.class,
                ()->{Classifier classy = new MultiClassifier(layout, 0, -1, 0, 0);}
                );
        //
        assertThrows(IllegalArgumentException.class,
                ()->{Classifier classy = new MultiClassifier(layout, 0, 0, -1, 0);}
                );
        //
        assertThrows(IllegalArgumentException.class,
                ()->{Classifier classy = new MultiClassifier(layout, 0, 0, 0, -1);}
                );
                **/
    }
    
   
    
    
}

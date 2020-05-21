package neural_network;


import java.io.*;
import java.util.*;
public class DataParser {

	/**
	 *
	 * @param filename - filepath pointing to a CSV format file
	 * @param num_attributes - number of attributes / input nodes
	 * @param num_classes - number of possible classification outcomes / output nodes
	 * @return
	 */
	public static Map<double[], double[]> readSupervisedInstances(String filename, int num_attributes, int num_classes){
		File file = new File(filename);
		Scanner sc;		
		Map<double[], double[]> instances = new HashMap<double[], double[]>();
		
		try {
			sc = new Scanner(file);		
			while(sc.hasNextLine()) {
				double[] instance = new double[num_attributes];
				String[] data = sc.nextLine().split("[,]");
				double[] classes = new double[num_classes];
				int i =0;
				for(i = 0; i<num_attributes; i++) {	//parse attribute values
					instance[i] = (double)Double.parseDouble(data[i]);					
				}				
				for(int j = 0; j<num_classes; j++) { //parse class identity values
					classes[j] = (double)Double.parseDouble(data[i]);	
					i++;
				}
				instances.put(instance, classes);
			}	
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch(ArrayIndexOutOfBoundsException e) {
			System.out.println("SYNTAX ERROR in "+filename+": The number of attributes or outputs at line "+ instances.size()+" is greater than number given by user");
			System.exit(0);
		}
		
		return instances;
	}
	
	public static Set<double[]> readUnsupervisedInstances(String filename, int num_attributes){
		File file = new File(filename);
		Scanner sc;
		Set<double[]> instances = new HashSet<double[]>();
		
		try {
			sc = new Scanner(file);			
			while(sc.hasNextLine()){
				double[] instance = new double[num_attributes];
				String[] data = sc.nextLine().split("[,]");
				for(int i = 0; i<num_attributes; i++) {	//parse attribute values
					instance[i] = (double)Double.parseDouble(data[i]);					
				}
				instances.add(instance);
			
			}
		}catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch(ArrayIndexOutOfBoundsException e) {
			System.out.println("SYNTAX ERROR in "+filename+": The number of attributes at line "+instances.size()+" is differs from the amount given ("+num_attributes+")");
			System.exit(0);
		}
		return instances;
	}
	
    public static List<Map<double[], double[]>> trainTestSplit(Map<double[], double[]> instances, double trainRatio){  
        assert(trainRatio >0 && trainRatio <= 1);
        
        List<Map<double[], double[]>> split = new ArrayList<Map<double[], double[]>>();
        Map<double[], double[]> train = new HashMap<double[], double[]>();
        Map<double[], double[]> test = new HashMap<double[], double[]>();      
        
        List<double[]> keys = new ArrayList<double[]>(instances.keySet());
        Collections.shuffle(keys);
        double len = keys.size();
        int trainCount = (int)(len * trainRatio);
        int i =0;
        for(double[] instance : keys) {
            if( i > trainCount) test.put(instance, instances.get(instance));
            else {
                train.put(instance, instances.get(instance));
            }
            i++;
        }
        split.add(train);
        split.add(test);
        return split;
    }
    

	

	


}

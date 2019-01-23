package neural_network;


import java.io.*;
import java.util.*;
public class DataParser {
	/**
	 * 
	 */
	
	
	

	/**
	 *
	 * 	
	 * 	
	 * 
	 *
	 * @param filename
	 * @param num_attributes
	 * @param num_classes Optional argument, used for training and test data sets. Value is equal to number of possible outputs/classes. 
	 * @throws FileNotFoundException
	 */
	public DataParser(){
		
	}
	
	
	public Map<double[], double[]> readSupervisedFile(String filename, int num_attributes, int num_classes){
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
				for(int j = 0; j<num_classes + num_attributes; j++) { //parse class identity values
					i++;
					classes[j] = (double)Double.parseDouble(data[i]);					
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
	
	public Set<double[]> readUnsupervisedFile(String filename, int num_attributes){
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
			System.out.println("SYNTAX ERROR in "+filename+": The number of attributes at line "+instances.size()+" is greater than number given by user ("+num_attributes+")");
			System.exit(0);
		}
		return instances;
	}
	

	


}

package neural_network;
import java.util.*;
public class Node {
			
	private double output;
	
	private ActivatedNode[] children;
	
	
	public Node(double output, ActivatedNode[] children) {
		this.output = output;		
		this.children = children;
	}
	
	
	public Node(double output,  int num_children) {
	    this.output = output;	    
	    this.children = new ActivatedNode[num_children];
	}
	
	public double getOutput() {
		return this.output;
	}
	
	public void setOutput(double output) {
		this.output = output;
	}
	
	public void setChildren(ActivatedNode[] children) {
		this.children = children;
	}
	
	public ActivatedNode[] getChildren() {
		return this.children;
	}
	
	
	
	
}

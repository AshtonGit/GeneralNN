package neural_network;
import java.util.*;
public class Node {
			
	private double output;
	private double error_signal;
	private ActivatedNode[] children;
	private static final int default_children = 5;
	//need an error signal 
	
	public Node(double output) {
		this.output = output;
		this.children = new ActivatedNode[default_children];
	}
	
	public Node(double output, ActivatedNode[] children) {
		this.output = output;
		this.children = children;
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
	
	public void setErrorSignal(double error_signal) {
		this.error_signal = error_signal;
	}
	
	public double getErrorSignal() {
		return this.error_signal;
	}
	
	
}

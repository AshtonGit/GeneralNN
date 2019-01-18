package neural_network;

import java.util.List;

public class Network {
	
	private List<Node[]> network;
	private double n_learn;
	private double dmax;
	private double momentum;
	private double flat_elim;
	
	
	public Network(List<Node[]> network, double n_learn, double dmax, double momentum, double flat_elim) {
		this.network = network;
		this.n_learn = n_learn;
		this.dmax = dmax;
		this.momentum = momentum;
		this.flat_elim = flat_elim;
	}
	
	
	public List<Node[]> getNodeNetwork(){
		return this.network;
	}
	
	public double getLearnRate() {
		return this.n_learn;
	}
	
	public double getMomentum() {
		return this.momentum;
	}
}

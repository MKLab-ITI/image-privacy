package modeling;

public class LinearModel {

	private String[] features;
	private double[] weights;

	public LinearModel(String[] features, double[] weights) {
		this.features = features;
		this.weights = weights;
	}

	public String[] getFeatures() {
		return features;
	}

	public void setFeatures(String[] features) {
		this.features = features;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}

}

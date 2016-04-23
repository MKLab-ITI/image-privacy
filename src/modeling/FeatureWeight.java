package modeling;

public class FeatureWeight {
	private String feature;
	private double weight;

	public FeatureWeight(String feature, double weight) {
		this.feature = feature;
		this.weight = weight;
	}

	public String getFeature() {
		return feature;
	}

	public void setFeature(String feature) {
		this.feature = feature;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

}

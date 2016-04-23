package modeling;

import java.util.ArrayList;

/**
 * This class can be used to replicate all the experimental results of the paper:<br>
 * 
 * E. Spyromitros-Xioufis, S. Papadopoulos, A. Popescu, Y. Kompatsiaris,
 * "Personalized Privacy-aware Image Classification", Proc. International Conference on Multimedia Retrieval
 * (ICMR), New York, USA, June 6-9, 2016.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class ExperimentsRunner {

	/** Path to the folder where the data sets reside */
	public static final String datasetFolder = "datasets/";

	public static void main(String[] args) throws Exception {
		genericExperiments();
		personalizedExperiments();
		insightsExperiments();
	}

	/**
	 * Generates the experimental results of Section 4.2 (Figures 3, 4, 5). More results are actually
	 * generated.
	 * 
	 * @throws Exception
	 */
	public static void genericExperiments() throws Exception {
		ArrayList<String> featureTypes = new ArrayList<String>();
		// -- Our features (available for both datasets) --
		featureTypes.add("vlad");
		featureTypes.add("cnn");
		featureTypes.add("semfeat");
		// -- Zerr et al. features (available only forPicAlert) --
		featureTypes.add("edch");
		featureTypes.add("bow");

		String classifier = "liblinear-tuned";

		int[] numTrainingExamples = { 50, 100, 500, 1000, 5000, -1 };

		for (int i = 0; i < numTrainingExamples.length; i++) {
			for (String featureType : featureTypes) {
				GenericModelEval.main(new String[] { datasetFolder, featureType, classifier,
						String.valueOf(numTrainingExamples[i]) });
			}
		}
	}

	/**
	 * Generates the experimental results of Section 4.3 (Figures 6, 7). More results are actually generated.
	 * 
	 * @throws Exception
	 */
	public static void personalizedExperiments() throws Exception {
		ArrayList<String> featureTypes = new ArrayList<String>();
		featureTypes.add("vlad");
		featureTypes.add("cnn");
		featureTypes.add("semfeat");
		// We use 'liblinear' (c=1) instead of 'liblinear-tuned' because it is impossible to tune the cost
		// parameter when only 5 or 10 examples are used for training in the 'user' models and we wanted to be
		// fair between 'user' models and the rest of the models.
		String classifier = "liblinear";

		int[] numUserSpecificExamples = { 5, 10, 15, 20, 25, 30, 35 };
		int[] hybridWeights = { 1, 10, 100, 1000 };

		String[] methods = { "generic", "other", "user", "hybrid-g", "hybrid-o" };

		for (String featureType : featureTypes) {
			System.out.println("=" + featureType);
			for (String method : methods) {
				System.out.println("==" + method);
				if (method.startsWith("hybrid") || method.equals("user")) {
					// repeat multiple times with different numbers of user-specific examples
					for (int j = 0; j < numUserSpecificExamples.length; j++) {
						if (method.equals("user")) {
							GenericAndPersonalModelEval.main(new String[] { datasetFolder, featureType,
									classifier, method + " " + String.valueOf(numUserSpecificExamples[j]) });
						} else {
							// repeat multiple times with different weights
							for (int k = 0; k < hybridWeights.length; k++) {
								GenericAndPersonalModelEval.main(new String[] { datasetFolder, featureType,
										classifier, method + " " + hybridWeights[k] + " "
												+ String.valueOf(numUserSpecificExamples[j]) });
							}
						}
					}
				} else if (method.equals("other") || method.startsWith("generic")) {
					GenericAndPersonalModelEval
							.main(new String[] { datasetFolder, featureType, classifier, method });
				} else {
					throw new Exception("Unknown method!");
				}
			}
		}
	}

	/**
	 * Generates the experimental results of Section 4.4. More results are actually generated.
	 * 
	 * @throws Exception
	 */
	public static void insightsExperiments() throws Exception {
		ModelExtraction.main(new String[] { datasetFolder + "youralert/semfeat.arff", "output/", "100" });
	}

}

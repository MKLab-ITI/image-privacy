package modeling;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import de.bwaldvogel.liblinear.Model;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class builds a (personalized) model for each user of YourAlert using only the examples of that user
 * and a (generic) model using all examples of YourAlert. All models are logistic regression from LibLinear.
 * For each model, the top positive (private) and negative (public) features (and the corresponding weights)
 * are extracted and written in txt files.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 */
public class ModelExtraction {

	/**
	 * @param args
	 *            [0] Full path to the semfeat version of the YourAlert dataset (e.g.
	 *            "datasets/youralert/semfeat.arff")
	 * @param args
	 *            [1] Full path to an output folder where model weights files will be written (e.g. "output/")
	 * @param args
	 *            [2] How many top private/public concepts to consider (e.g. "50")
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		String datasetPath = args[0];
		String outputPath = args[1];
		int topK = Integer.parseInt(args[2]);
		String classifierChoice = "liblinear";

		System.out.println("Loading YourAlert");
		BufferedReader reader = new BufferedReader(new FileReader(datasetPath));
		Instances data = new Instances(reader);
		data.setClassIndex(data.numAttributes() - 1); // set the class attribute
		reader.close();

		int numUsers = data.attribute(ConstantsAndUtils.userAttrIndex).numValues();
		// top positive and negative features are stored in HashMaps to ease discovery of interesting
		// deviations
		HashSet<String>[] posFeatures = new HashSet[numUsers + 1];
		HashSet<String>[] negFeatures = new HashSet[numUsers + 1];
		// build and evaluate a model for each YourAlert user
		for (int userIdIndex = 0; userIdIndex < numUsers; userIdIndex++) {
			String user = data.attribute(ConstantsAndUtils.userAttrIndex).value(userIdIndex);
			System.out.print("Building model for user: " + user);
			// filtering data that belong to other users
			Instances thisUserData = new Instances(data, 0);
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				if (inst.value(ConstantsAndUtils.userAttrIndex) == userIdIndex) {
					thisUserData.add(inst);
				}
			}
			System.out.println(", # examples: " + thisUserData.numInstances());
			// initialize a FilteredClassifier
			FilteredClassifier classifier = ConstantsAndUtils.createFilteredClassifier(
					ConstantsAndUtils.selectClassifier(classifierChoice), thisUserData,
					ConstantsAndUtils.indicesToIgnore);
			// build the model
			classifier.buildClassifier(thisUserData);
			thisUserData.delete(); // freeing resources
			System.gc();
			// extracting top features
			FeatureWeight[][] allFW = getTopFeatures(classifier, topK);
			FeatureWeight[] posFW = allFW[0];
			FeatureWeight[] negFW = allFW[1];
			// writing top features and weights in files
			BufferedWriter out = new BufferedWriter(
					new FileWriter(new File(outputPath + user + "-weights-private.txt")));
			out.write("concept,weight\n");
			out.write(topFeaaturesToCSVString(posFW));
			out.close();
			out = new BufferedWriter(new FileWriter(new File(outputPath + user + "-weights-public.txt")));
			out.write("concept,weight\n");
			out.write(topFeaaturesToCSVString(negFW));
			out.close();
			// populate hash maps for interesting deviation discovery
			posFeatures[userIdIndex] = new HashSet<String>(topK);
			for (int i = 0; i < topK; i++) {
				posFeatures[userIdIndex].add(posFW[i].getFeature());
			}
			negFeatures[userIdIndex] = new HashSet<String>(topK);
			for (int i = 0; i < topK; i++) {
				negFeatures[userIdIndex].add(negFW[i].getFeature());
			}

		}

		// build a generic model using examples from all users
		System.out.println("Building model on full dataset, # examples: " + data.numInstances());
		// initialize a FilteredClassifier
		FilteredClassifier classifier = ConstantsAndUtils.createFilteredClassifier(
				ConstantsAndUtils.selectClassifier(classifierChoice), data,
				ConstantsAndUtils.indicesToIgnore);
		// build the model
		classifier.buildClassifier(data);
		data.delete(); // freeing resources
		System.gc();
		// extracting top features
		FeatureWeight[][] allFW = getTopFeatures(classifier, topK);
		FeatureWeight[] posFW = allFW[0];
		FeatureWeight[] negFW = allFW[1];
		// writing features and weights in files
		BufferedWriter out = new BufferedWriter(
				new FileWriter(new File(outputPath + "generic-weights-private.txt")));
		out.write("concept,weight\n");
		out.write(topFeaaturesToCSVString(posFW));
		out.close();
		out = new BufferedWriter(new FileWriter(new File(outputPath + "generic-weights-public.txt")));
		out.write("concept,weight\n");
		out.write(topFeaaturesToCSVString(negFW));
		out.close();
		// populate hash maps for interesting deviation discovery
		posFeatures[numUsers] = new HashSet<String>(topK);
		for (int i = 0; i < topK; i++) {
			posFeatures[numUsers].add(posFW[i].getFeature());
		}
		negFeatures[numUsers] = new HashSet<String>(topK);
		for (int i = 0; i < topK; i++) {
			negFeatures[numUsers].add(negFW[i].getFeature());
		}

		// discover interesting deviations
		out = new BufferedWriter(new FileWriter(new File(outputPath + "deviations.txt")));
		out.write("\n===Interesting Deviations (considering top " + topK
				+ " private and public concepts) ===\n");
		for (int i = 0; i < (numUsers + 1); i++) { // for each model
			// check if its top positive features are in the top negative features of another model
			for (String feature : posFeatures[i]) {
				ArrayList<String> negativeUsers = new ArrayList<String>();
				for (int j = 0; j < (numUsers + 1); j++) {
					if (negFeatures[j].contains(feature)) {
						if (j == numUsers) {
							negativeUsers.add("generic");
						} else {
							negativeUsers.add(data.attribute(ConstantsAndUtils.userAttrIndex).value(j));
						}
					}
				}
				if (negativeUsers.size() > 0) {
					String user;
					if (i == numUsers) {
						user = "generic";
					} else {
						user = data.attribute(ConstantsAndUtils.userAttrIndex).value(i);
					}
					out.write("Concept: " + feature + " is private for model: " + user
							+ " and public for models: " + negativeUsers.toString() + "\n");
				}
			}
		}
		out.close();

	}

	/**
	 * Extracts the linear model and returns arrays of the top positive (private class) and negative (public
	 * class) features (concepts) and corresponding weights.
	 * 
	 * @param fc
	 * @param topK
	 * @return FeatureWeight[0] contains the top positive (private class) and FeatureWeight[1] contains the
	 *         top negative (public class) features (concepts) and corresponding weights.
	 * 
	 * @throws Exception
	 */
	public static FeatureWeight[][] getTopFeatures(Classifier fc, int topK) throws Exception {
		FeatureWeight[][] featuresAndWeights = new FeatureWeight[2][];
		LinearModel lm = extractLinearModel(fc);
		double[] weights = lm.getWeights();
		String[] features = lm.getFeatures();
		int[] sortedIndices = Utils.stableSort(weights);

		featuresAndWeights[0] = new FeatureWeight[topK]; // positive first
		for (int i = 0; i < topK; i++) {
			int posInOriginal = sortedIndices[sortedIndices.length - 1 - i];
			featuresAndWeights[0][i] = new FeatureWeight(features[posInOriginal], weights[posInOriginal]);
		}

		featuresAndWeights[1] = new FeatureWeight[topK]; // negative next
		for (int i = 0; i < topK; i++) {
			int posInOriginal = sortedIndices[i];
			featuresAndWeights[1][i] = new FeatureWeight(features[posInOriginal], weights[posInOriginal]);
		}
		return featuresAndWeights;
	}

	private static String topFeaaturesToCSVString(FeatureWeight[] fw) {
		int numDecimalPoints = 4;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < fw.length; i++) {
			double weight = fw[i].getWeight();
			String feature = fw[i].getFeature();
			sb.append(feature + "," + Utils.roundDouble(Math.abs(weight), numDecimalPoints) + "\n");
		}
		return sb.toString();
	}

	/**
	 * This method is used to extract the parameters of the logistic regression model build by LibLinear, when
	 * a FilteredClassifier that uses LibLinear as the classifier and Remove as the Filter is given.
	 * 
	 * @param fc
	 *            A FilteredClassifier with LibLINEAR as the classifier and Remove as the filter.
	 * @return
	 * @throws Exception
	 */
	private static LinearModel extractLinearModel(Classifier fc) throws Exception {
		// check if the classifier belongs to the LibLinear class
		Classifier classifier = ((FilteredClassifier) fc).getClassifier();
		Filter filter = ((FilteredClassifier) fc).getFilter();
		if (!(classifier instanceof LibLINEAR)) {
			throw new Exception("Method works only for LibLinear classifier!");
		}
		if (!(filter instanceof Remove)) {
			throw new Exception("Method works only for Remove filter!");
		}
		// get the model
		Model model = ((LibLINEAR) classifier).getModel();
		((LibLINEAR) classifier).getWeights();
		double[] weights = model.getFeatureWeights();
		// remove weight of the bias term + 1, due to a known bug in LIBLINEAR
		weights = Arrays.copyOfRange(weights, 0, weights.length - 2);
		// !!! RETURNED WEIGHTS HAVE INVERTED SIGNS (SOMETIMES) IN LIBLINEAR!!!
		int[] labels = model.getLabels();
		if (labels[0] == 0) { // invert signs to get correct weights
			for (int i = 0; i < weights.length; i++) {
				weights[i] = -weights[i];
			}
		}
		// get the attribute indices that are ignored by the filtered classifier
		Instances outputFormat = ((Remove) filter).getOutputFormat();
		// sanity check that the length of the weight vector is equal to the number of (non-ignored) features
		if (outputFormat.numAttributes() - 1 != weights.length) {
			throw new Exception("Expected weight vector length = " + (outputFormat.numAttributes() - 1) + ". "
					+ weights.length + " found!");
		}

		String[] features = new String[weights.length];
		int index = 0;
		for (int i = 0; i < outputFormat.numAttributes(); i++) {
			if (i != outputFormat.classIndex()) {
				features[index] = outputFormat.attribute(i).name();
				features[index] = prettyFormatSemfeat(features[index]);
				index++;
			}
		}
		return new LinearModel(features, weights);
	}

	private static String prettyFormatSemfeat(String name) {
		name = name.split("_", 2)[1].replace("_", "-");
		name = name.replace("0c", "youngster");
		return name;
	}
}

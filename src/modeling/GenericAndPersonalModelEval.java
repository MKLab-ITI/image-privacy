package modeling;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * This class is used for evaluating generic and personalized models on YourAlert. Both per user and average
 * performance results are reported.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 */
public class GenericAndPersonalModelEval {

	/**
	 * Num of fold used in k-fold cross-validation. See the paper for more details.
	 */
	public static final int numFolds = 10;
	/**
	 * The maximum number of generic examples that will be used for training.
	 */
	public static final int maxNumGenericExamples = 5000;

	/**
	 * 
	 * @param args
	 *            [0] Full path to the root folder where the PicAlert and YourAlert arffs reside. PicAlert
	 *            (YourAlert) arffs should be in the 'picalert' ('youralert') subfolder.
	 * @param args
	 *            [1] Name of the features to use in the evaluation (e.g. "semfeat")
	 * @param args
	 *            [2] Classifier selection (e.g. "liblinear")
	 * @param args
	 *            [3] The types of models to evaluate. Valid options are:<br>
	 *            <ul>
	 *            <li>generic: All YourAlert examples are predicted by a generic model trained on a random
	 *            sample of PicAlert.</li>
	 *            <li>other: The examples of each YourAlert user are predicted by a generic model trained on
	 *            all examples of the other YourAlert users.</li>
	 *            <li>user: The examples of each YourAlert user are predicted using personalized models that
	 *            are trained on subsets of the examples of that user.</li>
	 *            <li>hybrid-g: The examples of each YourAlert user are predicted using semi-personalized
	 *            models that are trained on a mixture of user-specific and generic examples (from PicAlert).
	 *            </li>
	 *            <li>hybrid-o: The examples of each YourAlert user are predicted using semi-personalized
	 *            models that are trained on a mixture of user-specific and generic examples (from other users
	 *            of YourAlert).</li>
	 *            </ul>
	 *            The last 3 types of models are evaluated using a modified k-fold cross-validation procedure
	 *            that is described in the paper.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		String datasetFolder = args[0];
		String featureType = args[1];
		String classifierChoice = args[2];
		String personalizationMethod = args[3];

		boolean loadPicAlert = false;
		String methodName = personalizationMethod.split(" ")[0];
		String methodNameCustom = "";
		int numUserSpecificExamples = 0; // hybrid/user-related parameter
		int userSpecificExamplesWeight = 0; // hybrid/user-related parameter
		if (methodName.startsWith("generic")) {
			// only examples from PicAlert will be used to create a single model
			loadPicAlert = true;
			methodNameCustom = methodName;
		} else if (methodName.equals("other")) {
			// only examples from other users of YourAlert will be used to predict each user
			methodNameCustom = methodName;
		} else if (methodName.equals("user")) {
			// only examples by the same user
			numUserSpecificExamples = Integer.parseInt(personalizationMethod.split(" ")[1]);
			userSpecificExamplesWeight = 1;
			methodNameCustom = methodName;
		} else if (methodName.startsWith("hybrid")) {
			// a combination of examples
			userSpecificExamplesWeight = Integer.parseInt(personalizationMethod.split(" ")[1]);
			numUserSpecificExamples = Integer.parseInt(personalizationMethod.split(" ")[2]);
			if (methodName.equals("hybrid-o")) {
				// do nothing
			} else if (methodName.equals("hybrid-g")) {
				loadPicAlert = true;
			} else {
				throw new Exception("Unknown method!");
			}
			methodNameCustom = methodName + " w=" + userSpecificExamplesWeight;
		} else {
			throw new Exception("Unknown method!");
		}

		Instances dataYouralert = null;
		File youralert = new File(datasetFolder + "youralert/" + featureType + ".arff");
		if (youralert.exists()) {
			System.out.println("Loading YourAlert");
			DataSource source = new DataSource(datasetFolder + "youralert/" + featureType + ".arff");
			dataYouralert = source.getDataSet();
			dataYouralert.setClassIndex(dataYouralert.numAttributes() - 1); // setting class attribute
		} else {
			throw new Exception("YourAlert dataset not found at:\n" + datasetFolder + "youralert/"
					+ featureType + ".arff");
		}

		Instances dataPicalert = null;
		if (loadPicAlert) {
			File picalert = new File(datasetFolder + "picalert/" + featureType + ".arff");
			if (picalert.exists()) {
				System.out.println("Loading PicAlert");
				DataSource source = new DataSource(datasetFolder + "picalert/" + featureType + ".arff");
				dataPicalert = source.getDataSet();
				dataPicalert.setClassIndex(dataPicalert.numAttributes() - 1); // setting class attribute
				// randomly picking maxNumGenericExamples examples from PicAlert
				if (dataPicalert.numInstances() > maxNumGenericExamples) {
					dataPicalert = pickAtRandom(dataPicalert, maxNumGenericExamples);
				}
			} else {
				throw new Exception("PicAlert dataset not found at:\n" + datasetFolder + "picalert/"
						+ featureType + ".arff");
			}
		}

		// create a file to write the results
		String resultsFile = "results-personal_" + featureType + "_" + personalizationMethod + "_"
				+ maxNumGenericExamples + "_" + classifierChoice + ".txt";
		BufferedWriter out = new BufferedWriter(new FileWriter(new File(resultsFile)));
		String constantOutput = featureType + "," + classifierChoice + "," + methodName + ","
				+ methodNameCustom + "," + userSpecificExamplesWeight + "," + maxNumGenericExamples + ","
				+ numUserSpecificExamples + "," + userSpecificExamplesWeight + ",";

		// build and evaluate a model for each user of YourAlert
		int numUsers = dataYouralert.attribute(ConstantsAndUtils.userAttrIndex).numValues();
		double[] mAucs = new double[numUsers];
		ArrayList<Prediction> allUsersPredictions = new ArrayList<Prediction>(dataYouralert.numInstances());

		for (int userIndex = 0; userIndex < numUsers; userIndex++) {
			String user = dataYouralert.attribute(ConstantsAndUtils.userAttrIndex).value(userIndex);
			System.out.println("Evaluation for user: " + user);
			// separate the YourAlert examples that belong to this user from the rest of the examples
			RemoveWithValues rwv = new RemoveWithValues();
			rwv.setAttributeIndex(String.valueOf(ConstantsAndUtils.userAttrIndex + 1));
			rwv.setNominalIndicesArr(new int[] { userIndex });
			rwv.setInvertSelection(true);
			rwv.setInputFormat(dataYouralert);
			rwv.setModifyHeader(false);
			Instances thisUserData = Filter.useFilter(dataYouralert, rwv);
			rwv.setInvertSelection(false);
			Instances otherUsersData = Filter.useFilter(dataYouralert, rwv);
			System.out.println("Examples of this user: " + thisUserData.numInstances());
			System.out.println("Remaining YourAlert examples " + otherUsersData.numInstances());

			// randomly picking maxNumGenericExamples examples from YourAlert
			if (otherUsersData.numInstances() > maxNumGenericExamples) {
				otherUsersData = pickAtRandom(otherUsersData, maxNumGenericExamples);
			}

			ArrayList<Prediction> thisUserPredictions = new ArrayList<Prediction>(
					thisUserData.numInstances());

			if (methodName.startsWith("generic") || methodName.equals("other")) {
				// the evaluation is simpler because all examples of this user can be predicted at one step

				// initialize a FilteredClassifier
				FilteredClassifier classifier = ConstantsAndUtils.createFilteredClassifier(
						ConstantsAndUtils.selectClassifier(classifierChoice), otherUsersData, ConstantsAndUtils.indicesToIgnore);

				// train and evaluate the model and store predictions
				System.out.println("Training");
				Instances trainingData = null;
				if (methodName.equals("other")) {
					trainingData = otherUsersData;
				} else if (methodName.equals("generic")) {
					trainingData = dataPicalert;
				}
				System.out.println("training with: " + trainingData.numInstances());
				classifier.buildClassifier(trainingData);
				Evaluation eval = new Evaluation(trainingData);
				System.out.println("Evaluation");
				eval.evaluateModel(classifier, thisUserData);
				thisUserPredictions = eval.predictions();
				allUsersPredictions.addAll(eval.predictions());
			} else if (methodName.equals("user") || methodName.startsWith("hybrid")) {
				// if data from this user are going to be used, the evaluation becomes more complex
				// prepare for stratified cv
				thisUserData.randomize(new Random(ConstantsAndUtils.seed)); // shuffle
				thisUserData.stratify(numFolds); // and stratify

				for (int n = 0; n < numFolds; n++) {
					System.out.println("Evaluation fold: " + n);

					Instances thisUserDataEvalFoldi = thisUserData.testCV(numFolds, n);
					Instances thisUserDataTrainFoldiInit = thisUserData.trainCV(numFolds, n);

					// pick the desired number of instances at random
					Instances thisUserDataTrainFoldiSample = pickAtRandom(thisUserDataTrainFoldiInit,
							numUserSpecificExamples);
					thisUserDataTrainFoldiInit.delete();

					Instances thisUserDataTrainFoldiSampleWeighted = new Instances(
							thisUserDataTrainFoldiSample, 0);
					// assign the appropriate weight
					for (int i = 0; i < thisUserDataTrainFoldiSample.numInstances(); i++) {
						Instance inst = thisUserDataTrainFoldiSample.instance(i);
						for (int k = 0; k < userSpecificExamplesWeight; k++) {
							thisUserDataTrainFoldiSampleWeighted.add(inst);
						}
					}
					thisUserDataTrainFoldiSample.delete();

					// create the final training set by combining the user-specific examples with generic
					// examples (depending on the method)
					Instances thisUserDataTrainFoldi = new Instances(dataYouralert, 0);
					if (methodName.equals("hybrid-o")) {
						thisUserDataTrainFoldi = new Instances(otherUsersData);
					} else if (methodName.equals("hybrid-g")) {
						thisUserDataTrainFoldi = new Instances(dataPicalert);
					}

					for (int i = 0; i < thisUserDataTrainFoldiSampleWeighted.numInstances(); i++) {
						thisUserDataTrainFoldi.add(thisUserDataTrainFoldiSampleWeighted.instance(i));
					}

					System.out.println(
							"Examples of this user for evaluation: " + thisUserDataEvalFoldi.numInstances());
					System.out.println(
							"Examples of this user for training: " + thisUserDataTrainFoldi.numInstances());

					// initializing a FilteredClassifier
					FilteredClassifier classifier = ConstantsAndUtils.createFilteredClassifier(
							ConstantsAndUtils.selectClassifier(classifierChoice), otherUsersData, ConstantsAndUtils.indicesToIgnore);
					// train
					classifier.buildClassifier(thisUserDataTrainFoldi);
					// evaluate
					Evaluation eval = new Evaluation(thisUserDataTrainFoldi);
					eval.evaluateModel(classifier, thisUserDataEvalFoldi);
					// add the predictions made for the examples of this fold
					thisUserPredictions.addAll(eval.predictions());
					allUsersPredictions.addAll(eval.predictions());
					thisUserDataEvalFoldi.delete();
					thisUserDataTrainFoldi.delete();
				}
			}

			// calculate the auc score based on all predictions
			ThresholdCurve tc = new ThresholdCurve();
			Instances result = tc.getCurve(thisUserPredictions, ConstantsAndUtils.privacyIndex);
			mAucs[userIndex] = ThresholdCurve.getROCArea(result);

			out.write(constantOutput + user + "," + mAucs[userIndex] + "\n");
			out.flush();

			otherUsersData.delete();
			thisUserData.delete();
		}

		// calculate the auc score based on all predictions
		ThresholdCurve tc = new ThresholdCurve();
		int privateIndex = 1;
		Instances result = tc.getCurve(allUsersPredictions, privateIndex);
		double genericAuc = ThresholdCurve.getROCArea(result);

		out.write(constantOutput + "average" + "," + genericAuc + "\n");

		out.close();

	}

	private static Instances pickAtRandom(Instances original, int numberToSelect) throws Exception {
		if (numberToSelect > original.numInstances()) {
			throw new Exception("Not enough instances!");
		}
		// check that both classes are represented
		if (numDistinctClassVals(original) < 2) {
			throw new Exception("All examples belong to the same class!");
		}
		ArrayList<Integer> indices = new ArrayList<Integer>(original.numInstances());
		for (int i = 0; i < original.numInstances(); i++) {
			indices.add(i);
		}
		Instances selected = null;
		int numClasses = 0;
		while (numClasses < 2) {
			Collections.shuffle(indices, new Random(ConstantsAndUtils.seed));
			selected = new Instances(original, 0);
			for (int i = 0; i < numberToSelect; i++) {
				selected.add(original.instance(indices.get(i)));
			}
			numClasses = numDistinctClassVals(selected);
		}
		return selected;
	}

	private static int numDistinctClassVals(Instances original) {
		HashSet<Double> distinctClassVals = new HashSet<Double>();
		for (int i = 0; i < original.numInstances(); i++) {
			distinctClassVals.add(original.instance(i).classValue());
		}
		return distinctClassVals.size();
	}
}

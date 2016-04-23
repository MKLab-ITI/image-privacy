package modeling;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * This class splits the PicAlert dataset randomly into train/test, builds a model on the train set and
 * evaluates it on the test set and on YourAlert.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class GenericModelEval {

	/** The percentage of PicAlert to use for training. */
	public static final double trainPercent = 60;

	/**
	 * 
	 * @param args
	 *            [0] Full path to the root folder where the PicAlert and YourAlert arffs reside. PicAlert
	 *            (YourAlert) arffs should be in the 'picalert' ('youralert') subfolder.
	 * @param args
	 *            [1] Name of the features to use in the evaluation (e.g. "semfeat")
	 * @param args
	 *            [2] Classifier selection (e.g. "liblinear-tuned")
	 * @param args
	 *            [3] How many generic training examples to use. < 0 means all examples
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		String datasetFolder = args[0];
		String featureType = args[1];
		String classifierChoice = args[2];
		int numTrainExamples = Integer.parseInt(args[3]);

		// check if the data sets exist in datasetFolder
		File picalert = new File(datasetFolder + "picalert/" + featureType + ".arff");
		Instances dataPicalert = null;
		if (picalert.exists()) {
			System.out.println("Loading PicAlert dataset");
			DataSource source = new DataSource(datasetFolder + "picalert/" + featureType + ".arff");
			dataPicalert = source.getDataSet();
			dataPicalert.setClassIndex(dataPicalert.numAttributes() - 1); // setting class attribute
			dataPicalert.randomize(new Random(ConstantsAndUtils.seed)); // randomly shuffle to discard any default
															// order
		} else {
			throw new Exception(
					"PicAlert dataset not found at:\n" + datasetFolder + "picalert/" + featureType + ".arff");
		}

		File youralert = new File(datasetFolder + "youralert/" + featureType + ".arff");
		Instances dataYouralert = null;
		if (youralert.exists()) {
			System.out.println("Loading YourAlert dataset");
			DataSource source = new DataSource(datasetFolder + "youralert/" + featureType + ".arff");
			dataYouralert = source.getDataSet();
			dataYouralert.setClassIndex(dataYouralert.numAttributes() - 1); // setting class attribute
		} else {
			System.err.println("YourAlert dataset not found at:\n" + datasetFolder + "youralert/"
					+ featureType + ".arff");
			System.err.println("Evaluation will be performed only on PicAlert!");
		}

		// create a file to write the evaluation results
		String resultsFilename = "results-generic_" + numTrainExamples + "_" + featureType + "_"
				+ classifierChoice + ".txt";
		BufferedWriter outResults = new BufferedWriter(new FileWriter(new File(resultsFilename)));
		String staticInfo = featureType + "," + numTrainExamples + "," + classifierChoice + ","
				+ numTrainExamples + ",";

		// initialize a FilteredClassifier
		FilteredClassifier classifier = ConstantsAndUtils.createFilteredClassifier(
				ConstantsAndUtils.selectClassifier(classifierChoice), dataPicalert, ConstantsAndUtils.indicesToIgnore);

		// split the PicAlert data set into train and test
		System.out.println("Splitting PicAlert dataset into train (" + trainPercent + "%) / test ("
				+ (100 - trainPercent) + "%)");
		Instances splitted[] = splitInTrainTest(dataPicalert, trainPercent, numTrainExamples);
		Instances picalertTrain = splitted[0];
		Instances picalertTest = splitted[1];
		System.out.println("Train examples " + picalertTrain.numInstances());
		System.out.println("Test instances " + picalertTest.numInstances());

		System.out.println("Building model on PicAlert train");
		Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
		copiedClassifier.buildClassifier(picalertTrain);

		System.out.println("Evaluating model on PicAlert test");
		Evaluation eval = new Evaluation(picalertTrain);
		eval.evaluateModel(copiedClassifier, picalertTest);
		double aucPicalert = eval.areaUnderROC(ConstantsAndUtils.privacyIndex);
		outResults.write(staticInfo + "all,picalert," + aucPicalert + "\n");

		if (dataYouralert != null) {
			System.out.println("Evaluating model on YourAlert");
			eval = new Evaluation(picalertTrain);

			// evaluate on all users
			eval.evaluateModel(copiedClassifier, dataYouralert);
			double aucYouralert = eval.areaUnderROC(ConstantsAndUtils.privacyIndex);
			outResults.write(staticInfo + "all,youralert," + aucYouralert + "\n");

			// evaluate separately per user
			int numUsers = dataYouralert.attribute(ConstantsAndUtils.userAttrIndex).numValues();
			for (int k = 0; k < numUsers; k++) {
				RemoveWithValues rwv = new RemoveWithValues();
				rwv.setAttributeIndex(String.valueOf(ConstantsAndUtils.userAttrIndex + 1));
				rwv.setNominalIndicesArr(new int[] { k });
				rwv.setInvertSelection(true);
				rwv.setInputFormat(dataYouralert);
				rwv.setModifyHeader(false);
				Instances dataThisUser = Filter.useFilter(dataYouralert, rwv);
				eval = new Evaluation(picalertTrain);
				eval.evaluateModel(copiedClassifier, dataThisUser);
				double aucThisUser = eval.areaUnderROC(ConstantsAndUtils.privacyIndex);
				String userName = dataYouralert.attribute(ConstantsAndUtils.userAttrIndex).value(k);
				outResults.write(staticInfo + userName + ",youralert," + aucThisUser + "\n");
			}
		}

		outResults.flush();
		outResults.close();
	}

	/**
	 * Splits the given data set into a training set (Instances[0]) and a test set (Instances[1]). The test
	 * set will contain (100-trainPercent)% of the examples, and training set will contain either the
	 * remaining examples or numTrainExamples if numTrainExamples > 0.
	 * 
	 * @param data
	 * @param trainPercent
	 *            The percentage of examples in the training set (between 1 and 100). The rest go to the test
	 *            set.
	 * @param numTrainExamples
	 *            If numTrainExamples > 0, the training set will be further subsampled (randomly) to contain
	 *            exactly numTrainExamples.
	 * @return
	 * @throws Exception
	 */
	public static Instances[] splitInTrainTest(Instances data, double trainPercent, int numTrainExamples)
			throws Exception {
		RemovePercentage rmvp = new RemovePercentage();
		rmvp.setInvertSelection(true);
		rmvp.setPercentage(trainPercent);
		rmvp.setInputFormat(data);
		Instances trainData = Filter.useFilter(data, rmvp);
		if (numTrainExamples > 0) { // remove examples until numTrainExamples are left
			while (trainData.numInstances() > numTrainExamples) {
				trainData.remove(0);
			}
		}
		rmvp = new RemovePercentage();
		rmvp.setPercentage(trainPercent);
		rmvp.setInputFormat(data);
		Instances testData = Filter.useFilter(data, rmvp);
		return new Instances[] { trainData, testData };

	}
}

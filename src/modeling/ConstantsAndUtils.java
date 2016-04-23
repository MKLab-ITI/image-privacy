package modeling;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.GridSearch9734Mod;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.AllFilter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class containing utility methods and variables.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class ConstantsAndUtils {

	/** The seed used for random number generation. */
	public static final int seed = 1;
	/**
	 * The first 3 attributes of the arff files (id, user, source) should be ignored when training.
	 */
	public static final String indicesToIgnore = "1-3";

	public static final int idAttrIndex = 0;
	public static final int userAttrIndex = 1;
	public static final int sourceAttrIndex = 2;
	public static final int privacyIndex = 1;

	public static Classifier selectClassifier(String choice) throws Exception {
		if (choice.equalsIgnoreCase("j48")) {
			J48 j48 = new J48();
			return j48;
		} else if (choice.equalsIgnoreCase("liblinear")) {
			LibLINEAR lib = new LibLINEAR();
			lib.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
			lib.setProbabilityEstimates(true);
			lib.setCost(1);
			return lib;
		} else if (choice.equals("liblinear-tuned")) {
			LibLINEAR liblinear = new LibLINEAR();
			liblinear.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
			liblinear.setProbabilityEstimates(true);

			GridSearch9734Mod grid = initializeGridSearch();
			grid.setFilter(new AllFilter()); // this filter is equal to not using a filter
			grid.setClassifier(liblinear);

			grid.setYProperty("classifier.cost");
			grid.setYMin(-2);
			grid.setYMax(2);
			grid.setYStep(1);
			grid.setYExpression("pow(BASE,I)");
			grid.setYBase(10);
			// below we use the default 1000 iterations, as well as a dummy value of 1 (to finish
			// quickly)
			grid.setXProperty("classifier.maximumNumberOfIterations");
			grid.setXMin(1);
			grid.setXMax(1000);
			grid.setXStep(999);
			grid.setXExpression("I");

			return grid;
		} else {
			throw new Exception("Wrong selection");
		}

	}

	/**
	 * Returns a FilteredClassifier that ignores specific attributes.
	 * 
	 * @param baseClassifier
	 * @param inputFormat
	 * @param indicesToIgnore
	 * @return
	 * @throws Exception
	 */
	public static FilteredClassifier createFilteredClassifier(Classifier baseClassifier,
			Instances inputFormat, String indicesToIgnore) throws Exception {
		FilteredClassifier filteredClassifier = new FilteredClassifier();
		filteredClassifier.setClassifier(baseClassifier);
		Remove rem = new Remove();
		rem.setAttributeIndices(indicesToIgnore);
		rem.setInputFormat(inputFormat);
		filteredClassifier.setFilter(rem);
		return filteredClassifier;
	}

	/**
	 * Initializes GridSearch with options that are common among all tunable classifiers. This is a modified
	 * version of Weka's GridSearch that allows setting the number of cv folds and does not repeat the tuning
	 * process.
	 * 
	 * @return
	 */
	private static GridSearch9734Mod initializeGridSearch() {
		GridSearch9734Mod grid = new GridSearch9734Mod();
		// the metric to optimize
		grid.setEvaluation(
				new SelectedTag(GridSearch9734Mod.EVALUATION_WAUC, GridSearch9734Mod.TAGS_EVALUATION));
		grid.setGridIsExtendable(false);
		grid.setNumExecutionSlots(2);
		grid.setSampleSizePercent(100);
		grid.setInitialNumFolds(2);
		grid.setStopAfterFirstGrid(true);
		grid.setTraversal(
				new SelectedTag(GridSearch9734Mod.TRAVERSAL_BY_ROW, GridSearch9734Mod.TAGS_TRAVERSAL));
		grid.setDebug(false);
		return grid;
	}
}

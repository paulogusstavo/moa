package moa.classifiers.drift;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.statisticaltests.KNN;
import moa.core.Measurement;

public class MDE extends AbstractClassifier implements MultiClassClassifier {
    protected Classifier classifier;
    protected Classifier[] ensemble;

    @Override
    public String getPurposeString() {
        return "MDE: Minority Driven Ensemble";
    }

    public IntOption ensemble_size = new IntOption("ensembleSize", 's',
            "Number of ensemble", 3,
            0, Integer.MAX_VALUE);

    public FloatOption alpha = new FloatOption("alpha", 'a',
            "Alpha", 0.05,
            0, Float.MAX_VALUE);

    public MultiChoiceOption decision = new MultiChoiceOption(
            "decision", 'd', "Decision", new String[] {
            "MIN", "BASIC"}, new String[] {"MIN", "BASIC"}, 0);

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return new double[0];
    }

    @Override
    public void resetLearningImpl() {
        this.classifier = (Classifier) new KNN().copy();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}

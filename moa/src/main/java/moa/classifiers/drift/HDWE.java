package moa.classifiers.drift;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;

public class HDWE extends AbstractClassifier implements MultiClassClassifier {

    private static final int n_estimators = 10;
    private static final int n_splits = 5;
    private static final String pred_type = "soft";

    @Override
    public String getPurposeString() {
        return "HDWE: Hellinger Distance Weighted Ensemble.";
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {


        return new double[0];
    }

    @Override
    public void resetLearningImpl() {

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

package ffx.potential.constraint;

import ffx.numerics.Constraint;

import static ffx.utilities.Constants.KCAL_TO_GRAM_ANG2_PER_PS2;

public class ShakeChargeConstraint implements Constraint {
    final int nConstraints;
    final double tol;
    final int c;
    final double maxIters =150;

    public ShakeChargeConstraint(int nConstraints, int c, double tol){
        this.nConstraints = nConstraints;
        this.c = c;
        this.tol = tol;
    }
    @Override
    public void applyConstraintToStep(final double[] xPrior, double[] xNew, final double[] masses, double tol){
    }

    public void applyChargeConstraintToStep(final double[] x, final double[] afric, final double[] masses, final double dt){
        boolean done = false;
        int iter = 0;
        while(!done){
            done = true;
            iter++;
            double totalLambda = 0.0;
            double totalInverseMass = 0.0;
            for (int i = 0; i < nConstraints; i++) {
                double lambda = Math.sin(x[i]) * Math.sin(x[i]);
                totalLambda += lambda;
                totalInverseMass += (1.0 / masses[i]);
            }
            double delta = totalLambda - c;

            if (Math.abs(delta) > tol) {
                done = false;
                double term = delta / (dt * dt * totalInverseMass);
                for (int i = 0; i < nConstraints; i++) {
                    double g = -term * Math.sin(2 * x[i]);
                    x[i] = x[i] + g * afric[i];
                }
            }
            if(iter >= maxIters){
                throw new RuntimeException("SHAKE Charge Constraint -- Warning, Charge Constraint not Satisfied");
            }
        }
    }

    @Override
    public void applyConstraintToVelocities(final double[] x, double[] v, final double[] masses, double tol){

    }

    @Override
    public int[] constrainedAtomIndices() {
        return new int[0];
    }

    @Override
    public boolean constraintSatisfied(double[] x, double tol) {
        return false;
    }

    @Override
    public boolean constraintSatisfied(double[] x, double[] v, double xTol, double vTol) {
        return false;
    }

    @Override
    public int getNumDegreesFrozen() {
        return 0;
    }
}



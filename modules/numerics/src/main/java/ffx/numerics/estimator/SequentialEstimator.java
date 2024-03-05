// ******************************************************************************
//
// Title:       Force Field X.
// Description: Force Field X - Software for Molecular Biophysics.
// Copyright:   Copyright (c) Michael J. Schnieders 2001-2024.
//
// This file is part of Force Field X.
//
// Force Field X is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
// Force Field X is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// Force Field X; if not, write to the Free Software Foundation, Inc., 59 Temple
// Place, Suite 330, Boston, MA 02111-1307 USA
//
// Linking this library statically or dynamically with other modules is making a
// combined work based on this library. Thus, the terms and conditions of the
// GNU General Public License cover the whole combination.
//
// As a special exception, the copyright holders of this library give you
// permission to link this library with independent modules to produce an
// executable, regardless of the license terms of these independent modules, and
// to copy and distribute the resulting executable under terms of your choice,
// provided that you also meet, for each linked independent module, the terms
// and conditions of the license of that module. An independent module is a
// module which is not derived from or based on this library. If you modify this
// library, you may extend this exception to your version of the library, but
// you are not obligated to do so. If you do not wish to do so, delete this
// exception statement from your version.
//
// ******************************************************************************
package ffx.numerics.estimator;

import java.util.ArrayList;

import static java.lang.System.arraycopy;
import static java.util.Arrays.copyOf;
import static java.util.Arrays.fill;
import static java.util.Arrays.stream;

/**
 * The SequentialEstimator abstract class defines a statistical estimator based on perturbative
 * potential energy differences between adjacent windows (e.g. exponential free energy perturbation,
 * Bennett Acceptance Ratio, etc).
 *
 * @author Michael J. Schnieders
 * @author Jacob M. Litman
 * @since 1.0
 */
public abstract class SequentialEstimator implements StatisticalEstimator {

  protected final double[] lamValues;
  protected final double[][] eLow;
  protected final double[][] eAt;
  protected final double[][] eHigh;
  protected final double[][][] eAll; // [[[energies]perturbationsAcrossLambda]lambdaWindow] N X K array for each lambda window
  protected final double[][] eAllFlat; // [lambda][evaluationsAtLambda]
  protected final double[] temperatures;
  protected final int nTrajectories;

  /**
   * The SequentialEstimator constructor largely just copies its parameters into local variables.
   * Most arrays are duplicated (rather a just copying their reference).
   * The temperature array can be of length 1 if all elements are meant to be the same temperature.
   *
   * <p>The first dimension of the energies arrays corresponds to the lambda values/windows. The
   * second dimension (can be of uneven length) corresponds to potential energies of snapshots
   * sampled from that lambda value, calculated either at that lambda value, the lambda value below,
   * or the lambda value above. The arrays energiesLow[0] and energiesHigh[n-1] is expected to be all
   * NaN.
   *
   * @param lambdaValues Values of lambda dynamics was run at.
   * @param energiesLow Potential energies of trajectory L at lambda L-dL.
   * @param energiesAt Potential energies of trajectory L at lambda L.
   * @param energiesHigh Potential energies of trajectory L at lambda L+dL.
   * @param temperature Temperature each lambda window was run at (single-element indicates
   *     identical temperatures).
   */
  public SequentialEstimator(double[] lambdaValues, double[][] energiesLow, double[][] energiesAt,
      double[][] energiesHigh, double[] temperature) {
    nTrajectories = lambdaValues.length;
    eAll = null;
    eAllFlat = null;

    assert stream(energiesLow[0]).allMatch(Double::isNaN)
        && stream(energiesHigh[nTrajectories - 1]).allMatch(Double::isNaN);

    assert nTrajectories == energiesAt.length
        && nTrajectories == energiesLow.length
        && nTrajectories == energiesHigh.length
        : "One of the energy arrays is of the incorrect length in the first dimension!";

    this.lamValues = copyOf(lambdaValues, nTrajectories);
    temperatures = new double[nTrajectories];
    if (temperature.length == 1) {
      fill(temperatures, temperature[0]);
    } else {
      arraycopy(temperature, 0, temperatures, 0, nTrajectories);
    }

    // Just in case, copy the arrays rather than storing them as provided.
    eLow = new double[nTrajectories][];
    eAt = new double[nTrajectories][];
    eHigh = new double[nTrajectories][];
    for (int i = 0; i < nTrajectories; i++) {
      eLow[i] = copyOf(energiesLow[i], energiesLow[i].length);
      eAt[i] = copyOf(energiesAt[i], energiesAt[i].length);
      eHigh[i] = copyOf(energiesHigh[i], energiesHigh[i].length);
    }
  }


  /**
   * The SequentialEstimator constructor largely just copies its parameters into local variables.
   * Most arrays are duplicated (rather a just copying their reference).
   * The temperature array can be of length 1 if all elements are meant to be the same temperature.
   *<p>
   * This constructor is meant for lower variance estimators such as MBAR & WHAM. These methods require energy
   * evaluations from all lambda windows at all lambda values. The energiesAll array is expected to be
   * of the form energiesAll[lambdaWindow][windowPerspective][lambdaWindowSnapshotPerspectiveEnergy].
   * As an example, at the 3rd lambda window, the energiesAll[2] array should contain the energies
   * of all the snapshots from the 3rd lambda window evaluated at all lambda values. energiesAll[2][3] is a
   * list of all snapshots from lambda 3 evaluated with the potential of lambda 4. energiesAll[2][3][4] is
   * the 5th snapshot from lambda 3 evaluated with the potential of lambda 4.
   * <p>
   * This constructor also breaks energiesAll into a flattened array (across the second dimension) such that
   * the first dimension is the lambda window where the energy was evaluated and the second dimension is the
   * samples. energiesAll is also broken down into eAt, eLow, and eHigh arrays for convenience & so that BAR
   * calculations can be performed and compared.
   *
   * @param lambdaValues Values of lambda dynamics was run at.
   * @param energiesAll Potential energies of trajectory L at all other lambdas.
   * @param temperature Temperature each lambda window was run at (single-element indicates
   *                    identical temperatures).
   */
  public SequentialEstimator(double[] lambdaValues, double[][][] energiesAll, double[] temperature) {
    nTrajectories = lambdaValues.length;

    assert nTrajectories == energiesAll.length
        : "The energy arrays is of the incorrect length in the first lambda dimension!";

    assert nTrajectories == energiesAll[0].length
        : "The energy arrays is of the incorrect length in the second lambda dimension!";

    this.lamValues = copyOf(lambdaValues, nTrajectories);
    temperatures = new double[nTrajectories];
    if (temperature.length == 1) {
      fill(temperatures, temperature[0]);
    } else {
      arraycopy(temperature, 0, temperatures, 0, nTrajectories);
    }

    // Just in case, deep copy the array rather than storing them as provided.
    eAll = new double[nTrajectories][][];
    for (int i = 0; i < nTrajectories; i++) {
      eAll[i] = new double[energiesAll[i].length][];
      for (int j = 0; j < energiesAll[i].length; j++) {
        eAll[i][j] = copyOf(energiesAll[i][j], energiesAll[i][j].length);
      }
    }

    eAllFlat = new double[nTrajectories][];
    for (int i = 0; i < nTrajectories; i++) {
        ArrayList<Double> temp = new ArrayList<>();
        for(int j = 0; j < nTrajectories; j++) {
          for(int k = 0; k < eAll[j][i].length; k++) {
            temp.add(eAll[j][i][k]);
          }
        }
        eAllFlat[i] = temp.stream().mapToDouble(Double::doubleValue).toArray();
    }

    // Assert that lengths of the energiesAll arrays are correct.
    for(int i = 0; i < nTrajectories; i++){
      assert eAll[i].length == nTrajectories :
              "The energy arrays is of the incorrect length in the second lambda dimension at lambda " + i + "!";
      int nSnapshots = eAll[i][0].length;
        for(int j = 0; j < nTrajectories; j++){
            assert eAll[i][j].length == nSnapshots :
                    "The energy arrays is of the incorrect length in numSnaps dimension at lambda " +
                            i + " for evaluation at lambda " + j + "!";
        }
    }

    // Initialize the eLow, eAt, and eHigh arrays to their expected values from eAll.
    eLow = new double[nTrajectories][eAll[0][0].length];
    fill(eLow[0], Double.NaN);
    eAt = new double[nTrajectories][];
    eHigh = new double[nTrajectories][eAll[0][0].length];
    fill(eHigh[nTrajectories - 1], Double.NaN);
    for (int i = 0; i < nTrajectories; i++) {
      if (i != 0) {
        eLow[i] = copyOf(eAll[i][i-1], eAll[i][i-1].length);
      }
      eAt[i] = copyOf(eAll[i][i], eAll[i][i].length);
      if(i != nTrajectories - 1) {
        eHigh[i] = copyOf(eAll[i][i + 1], eAll[i][i + 1].length);
      }
    }
  }
}

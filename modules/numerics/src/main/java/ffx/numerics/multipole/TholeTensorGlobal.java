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
package ffx.numerics.multipole;

import java.util.Arrays;

import static ffx.numerics.special.Erf.erf;
import static ffx.numerics.special.Erf.erfc;
import static java.lang.Math.PI;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * The TholeTensorGlobal class computes derivatives of Thole damping via recursion to order &lt;= 4 for
 * Cartesian multipoles in either a global frame.
 *
 * @author Michael J. Schnieders
 * @see <a href="http://doi.org/10.1142/9789812830364_0002" target="_blank"> Matt Challacombe, Eric
 *     Schwegler and Jan Almlof, Modern developments in Hartree-Fock theory: Fast methods for
 *     computing the Coulomb matrix. Computational Chemistry: Review of Current Trends. pp. 53-107,
 *     Ed. J. Leczszynski, World Scientifc, 1996. </a>
 * @since 1.0
 */
public class TholeTensorGlobal extends CoulombTensorGlobal {

  /** Constant <code>threeFifths=3.0 / 5.0</code> */
  private static final double threeFifths = 3.0 / 5.0;

  /** Constant <code>oneThirtyFifth=1.0 / 35.0</code> */
  private static final double oneThirtyFifth = 1.0 / 35.0;

  /**
   * Thole damping parameter is set to min(pti,ptk)).
   */
  private double thole;

  /**
   * AiAk parameter = 1/(alphaI^6*alphaK^6) where alpha is polarizability.
   */
  private double AiAk;
  double[] tholeSource;
  double beta;
  private static final double sqrtPI = sqrt(PI);

  /**
   * Constructor for EwaldMultipoleTensorGlobal.
   *
   * @param order Tensor order.
   * @param thole Thole damping parameter is set to min(pti,ptk)).
   * @param AiAk parameter = 1/(alphaI^6*alphaK^6) where alpha is polarizability.
   */
  public TholeTensorGlobal(int order, double thole, double AiAk) {
    super(order);
    this.thole = thole;
    this.AiAk = AiAk;
    this.operator = Operator.THOLE_FIELD;

    // Source terms are currently defined up to order 4.
    //assert (order <= 4);
    tholeSource = new double[o1];
  }

  private void initTholeSource(int order, double R, double thole, double aiAk, double[] tholeSource) {
    double R2 = R * R;
    beta = thole*aiAk*R2;
    double prefactor = 2.0 * beta / sqrtPI;
    double twoBeta2 = -2.0 * beta * beta;
    for (int n = 0; n <= order; n++) {
      tholeSource[n] = prefactor * pow(twoBeta2, n);
    }
  }

  /**
   * Set Thole damping parameters
   *
   * @param thole a double.
   * @param AiAk a double.
   */
  public void setThole(double thole, double AiAk) {
    this.thole = thole;
    this.AiAk = AiAk;
  }

  /**
   * Check if the Thole damping is exponential is greater than zero (or the interaction can be
   * neglected).
   *
   * @param r The separation distance.
   * @return True if -thole*u^3 is greater than -50.0.
   */
  public boolean checkThole(double r) {
    return checkThole(thole, AiAk, r);
  }

  /**
   * Check if the Thole damping is exponential is greater than zero (or the interaction can be
   * neglected).
   *
   * @param thole Thole damping parameter is set to min(pti,ptk)).
   * @param AiAk parameter = 1/(alphaI^6*alphaK^6) where alpha is polarizability.
   * @param r The separation distance.
   * @return True if -thole*u^3 is greater than -50.0.
   */
  protected static boolean checkThole(double thole, double AiAk, double r) {
    double rAiAk = r * AiAk;
    return (-thole * rAiAk * rAiAk * rAiAk > -50.0);
  }

  /**
   * Generate source terms for the Challacombe et al. recursion.
   *
   * @param T000 Location to store the source terms.
   */
  @Override
  protected void source(double[] T000) {
    // Compute the normal Coulomb auxiliary term.
    //super.source(T000);

    // Add the Thole damping terms: edamp = exp(-thole*u^3).
    initTholeSource(order, R, thole, AiAk, tholeSource);
    tholeSource(o1 - 1, beta, R, tholeSource, T000);
  }

  /**
   * Generate source terms for the Challacombe et al. recursion.
   *
   * @param R The separation distance.
   * @param T000 Location to store the source terms.
   */
  protected static void tholeSource(int order, double beta, double R, double[] source, double[] T000) {
    double betaR = beta * R;
    double betaR2 = betaR * betaR;
    double iBetaR2 = 1.0 / (2.0 * betaR2);
    double expBR2 = exp(-betaR2);
    // Fn(x^2) = Sqrt(PI) * erf(x) / (2*x)
    // where x = R
    double Fn = sqrtPI * erf(R) / (2.0 * betaR);
    for (int n = 0; n <= order; n++) {
      T000[n] = source[n] * Fn;
      // Generate F(n+1)c from Fnc (Eq. 2.24 in Sagui et al.)
      // F(n+1)c = [(2*n+1) Fnc(x) + exp(-x)] / 2x
      // where x = (Beta*R)^2
      Fn = ((2.0 * n + 1.0) * Fn - expBR2) * iBetaR2;
    }
  }
}

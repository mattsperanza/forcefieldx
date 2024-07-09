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
package ffx.numerics.fft;

import java.util.Random;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;

import static java.lang.Math.fma;
import static java.lang.System.arraycopy;
import static org.apache.commons.math3.util.FastMath.PI;
import static org.apache.commons.math3.util.FastMath.cos;
import static org.apache.commons.math3.util.FastMath.sin;

/**
 * Compute the FFT of complex, double precision data of arbitrary length n. This class uses a mixed
 * radix method and has special methods to handle factors [2, 3, 4, 5, 6, 7] and a general method for
 * larger prime factors.
 *
 * @author Michal J. Schnieders<br>
 * Derived from:
 * <br>
 * Bruce R. Miller (bruce.miller@nist.gov)
 * <br>
 * Contribution of the National Institute of Standards and Technology, not subject to copyright.
 * <br>
 * Derived from:
 * <br>
 * GSL (Gnu Scientific Library) FFT Code by Brian Gough (bjg@network-theory.co.uk)
 * @see <ul>
 * <li><a href="http://dx.doi.org/10.1016/0021-9991(83)90013-X" target="_blank"> Clive
 * Temperton. Self-sorting mixed-radix fast fourier transforms. Journal of Computational
 * Physics, 52(1):1-23, 1983. </a>
 * <li><a href="http://www.jstor.org/stable/2003354" target="_blank"> J. W. Cooley and J. W.
 * Tukey, Mathematics of Computation 19 (90), 297 (1965) </a>
 * <li><a href="http://en.wikipedia.org/wiki/Fast_Fourier_transform" target="_blank">FFT at Wikipedia </a>
 * </ul>
 * @since 1.0
 */
public class Complex {

  private static final Logger logger = Logger.getLogger(Complex.class.getName());
  // TINKER v. 5.0 factors to achieve exact numerical agreement.
  // private static final int[] availableFactors = {5, 4, 3, 2};
  // private static final int firstUnavailablePrime = 7;
  private static final int[] availableFactors = {7, 6, 5, 4, 3, 2};
  private static final int firstUnavailablePrime = 11;
  /**
   * Number of complex numbers in the transform.
   */
  private final int n;
  /**
   * Factorization of n.
   */
  private final int[] factors;
  /**
   * Twiddle factors.
   */
  private final double[][][] twiddle;
  /**
   * Packing of non-contiguous data.
   */
  private final double[] packedData;
  /**
   * Scratch space for the transform.
   */
  private final double[] scratch;
  /**
   * References to the input and output data arrays.
   */
  private final PassData[] passData;
  /**
   * Constants for each pass.
   */
  private final MixedRadixFactor[] mixedRadixFactors;

  private boolean useSIMD;

  /**
   * Construct a Complex instance for data of length n. Factorization of n is designed to use special
   * methods for small factors, and a general routine for large odd prime factors. Scratch memory is
   * created of length 2*n, which is reused each time a transform is computed.
   *
   * @param n Number of complex numbers (n .GT. 1).
   */
  public Complex(int n) {
    assert (n > 1);
    this.n = n;
    factors = factor();
    twiddle = wavetable();
    packedData = new double[2 * n];
    scratch = new double[2 * n];
    passData = new PassData[2];

    // Create mixed radix instances for each factor.
    mixedRadixFactors = new MixedRadixFactor[factors.length];
    int product = 1;
    for (int i = 0; i < factors.length; i++) {
      final int factor = factors[i];
      product *= factor;
      final int outerLoopLimit = n / product;
      final int innerLoopLimit = product / factor;
      final int nextInput = n / factor;
      final int di = 2 * nextInput;
      final int dj = 2 * innerLoopLimit;
      PassConstants passConstants = new PassConstants(factor, product, outerLoopLimit, innerLoopLimit, nextInput, di, dj, twiddle[i]);
      switch (factor) {
        case 2 -> mixedRadixFactors[i] = new MixedRadixFactor2(passConstants);
        case 3 -> mixedRadixFactors[i] = new MixedRadixFactor3(passConstants);
        case 4 -> mixedRadixFactors[i] = new MixedRadixFactor4(passConstants);
        case 5 -> mixedRadixFactors[i] = new MixedRadixFactor5(passConstants);
        case 6 -> mixedRadixFactors[i] = new MixedRadixFactor6(passConstants);
        case 7 -> mixedRadixFactors[i] = new MixedRadixFactor7(passConstants);
        default -> mixedRadixFactors[i] = new MixedRadixFactorPrime(passConstants);
      }
    }

    // Use SIMD by default only for AVX-512.
    useSIMD = MixedRadixFactor.SPECIES_LENGTH == 8;
    String simd = System.getProperty("fft.useSIMD", Boolean.toString(useSIMD));
    try {
      useSIMD = Boolean.parseBoolean(simd);
    } catch (Exception e) {
      logger.info(" Invalid value for fft.useSIMD: " + simd);
      useSIMD = false;
    }
  }

  /**
   * Configure use of SIMD operators.
   *
   * @param useSIMD True to use SIMD operators.
   */
  public void setUseSIMD(boolean useSIMD) {
    this.useSIMD = useSIMD;
  }

  /**
   * Check if a dimension is a preferred dimension.
   *
   * @param dim the dimension to check.
   * @return true if the dimension is a preferred dimension.
   */
  public static boolean preferredDimension(int dim) {
    if (dim < 2) {
      return false;
    }

    // Apply preferred factors.
    for (int factor : availableFactors) {
      while ((dim % factor) == 0) {
        dim /= factor;
      }
    }
    return dim <= 1;
  }

  /**
   * Getter for the field <code>factors</code>.
   *
   * @return an array of int.
   */
  public int[] getFactors() {
    return factors;
  }

  /**
   * Compute the Fast Fourier Transform of data leaving the result in data. The array data must
   * contain the data points in the following locations:
   *
   * <PRE>
   * Re(d[i]) = data[offset + stride*i]
   * Im(d[i]) = data[offset + stride*i+1]
   * </PRE>
   *
   * @param data   an array of double.
   * @param offset the offset to the beginning of the data.
   * @param stride the stride between data points.
   */
  public void fft(double[] data, int offset, int stride) {
    transformInternal(data, offset, stride, -1);
  }

  /**
   * Compute the (un-normalized) inverse FFT of data, leaving it in place. The frequency domain data
   * must be in wrap-around order, and be stored in the following locations:
   *
   * <PRE>
   * Re(D[i]) = data[offset + stride*i]
   * Im(D[i]) = data[offset + stride*i+1]
   * </PRE>
   *
   * @param data   an array of double.
   * @param offset the offset to the beginning of the data.
   * @param stride the stride between data points.
   */
  public void ifft(double[] data, int offset, int stride) {
    transformInternal(data, offset, stride, +1);
  }

  /**
   * Compute the normalized inverse FFT of data, leaving it in place. The frequency domain data must
   * be stored in the following locations:
   *
   * <PRE>
   * Re(D[i]) = data[offset + stride*i]
   * Im(D[i]) = data[offset + stride*i+1]
   * </PRE>
   *
   * @param data   an array of double.
   * @param offset the offset to the beginning of the data.
   * @param stride the stride between data points.
   */
  public void inverse(double[] data, int offset, int stride) {
    ifft(data, offset, stride);

    // Normalize inverse FFT with 1/n.
    double norm = normalization();
    for (int i = 0; i < n; i++) {
      final int index = offset + stride * i;
      data[index] *= norm;
      data[index + 1] *= norm;
    }
  }

  /**
   * Factor the data length into preferred factors (those with special methods), falling back to odd
   * primes that the general routine must handle.
   *
   * @return integer factors
   */
  private int[] factor() {
    if (n < 2) {
      return null;
    }
    Vector<Integer> v = new Vector<>();
    int nTest = n;

    // Use the preferred factors first
    for (int factor : availableFactors) {
      while ((nTest % factor) == 0) {
        nTest /= factor;
        v.add(factor);
      }
    }

    // Unavailable odd prime factors.
    int factor = firstUnavailablePrime;
    while (nTest > 1) {
      while ((nTest % factor) != 0) {
        factor += 2;
      }
      nTest /= factor;
      v.add(factor);
    }
    int product = 1;
    int nf = v.size();
    int[] ret = new int[nf];
    for (int i = 0; i < nf; i++) {
      ret[i] = v.get(i);
      product *= ret[i];
    }

    // Report a failed factorization.
    if (product != n) {
      StringBuilder sb = new StringBuilder(" FFT factorization failed for " + n + "\n");
      for (int i = 0; i < nf; i++) {
        sb.append(" ");
        sb.append(ret[i]);
      }
      sb.append("\n");
      sb.append(" Factor product = ");
      sb.append(product);
      sb.append("\n");
      logger.severe(sb.toString());
      System.exit(-1);
    } else {
      if (logger.isLoggable(Level.FINEST)) {
        StringBuilder sb = new StringBuilder(" FFT factorization for " + n + " = ");
        for (int i = 0; i < nf - 1; i++) {
          sb.append(ret[i]);
          sb.append(" * ");
        }
        sb.append(ret[nf - 1]);
        logger.finest(sb.toString());
      }
    }
    return ret;
  }

  /**
   * Compute the Fast Fourier Transform of data leaving the result in data.
   *
   * @param data   data an array of double.
   * @param offset the offset to the beginning of the data.
   * @param stride the stride between data points.
   * @param sign   the sign to apply (forward -1 and inverse 1).
   */
  private void transformInternal(final double[] data, final int offset, final int stride, final int sign) {

    boolean packed = false;
    if (stride != 2) {
      // Pack non-contiguous (stride > 2) data into a contiguous array.
      packed = true;
      for (int i = 0, i2 = 0, index = offset; i < n; i++, i2 += 2, index += stride) {
        packedData[i2] = data[index];
        packedData[i2 + 1] = data[index + 1];
      }
      passData[0] = new PassData(sign, packedData, 0, scratch, 0);
      passData[1] = new PassData(sign, scratch, 0, packedData, 0);
    } else {
      passData[0] = new PassData(sign, data, offset, scratch, 0);
      passData[1] = new PassData(sign, scratch, 0, data, offset);
    }

    final int nfactors = factors.length;
    for (int i = 0; i < nfactors; i++) {
      final int pass = i % 2;
      if (useSIMD) {
        mixedRadixFactors[i].passSIMD(passData[pass]);
      } else {
        mixedRadixFactors[i].pass(passData[pass]);
      }
    }

    // If the number of factors is odd, the final result is in the scratch array.
    if (nfactors % 2 == 1) {
      // Copy the scratch array to the data array.
      if (stride == 2) {
        arraycopy(scratch, 0, data, offset, 2 * n);
      } else {
        for (int i = 0, i2 = 0, index = offset; i < n; i++, i2 += 2, index += stride) {
          data[index] = scratch[i2];
          data[index + 1] = scratch[i2 + 1];
        }
      }
      // If the number of factors is even, the data may need to be unpacked.
    } else if (packed) {
      for (int i = 0, i2 = 0, index = offset; i < n; i++, i2 += 2, index += stride) {
        data[index] = packedData[i2];
        data[index + 1] = packedData[i2 + 1];
      }
    }
  }

  /**
   * Return the normalization factor. Multiply the elements of the back-transformed data to get the
   * normalized inverse.
   *
   * @return a double.
   */
  private double normalization() {
    return 1.0 / n;
  }

  /**
   * Compute twiddle factors. These are trigonometric constants that depend on the factoring of n.
   *
   * @return twiddle factors.
   */
  private double[][][] wavetable() {
    if (n < 2) {
      return null;
    }

    // System.out.println(" Computing twiddle factors for length :" + n);
    // System.out.println(" Number of factors: " + factors.length);

    final double TwoPI_N = -2.0 * PI / n;
    final double[][][] ret = new double[factors.length][][];
    int product = 1;
    for (int i = 0; i < factors.length; i++) {
      int factor = factors[i];
      int product_1 = product;
      product *= factor;
      // The number of twiddle factors for the current factor.
      int outLoopLimit = n / product;
      // For the general odd pass, we need to add 1.
      if (factor >= firstUnavailablePrime) {
        outLoopLimit += 1;
      }
      final int nTwiddle = factor - 1;
      // System.out.println("\n  Factor: " + factor);
      // System.out.println("   Product: " + product);
      ret[i] = new double[outLoopLimit][2 * nTwiddle];
      // System.out.printf("   Size: T(%d,%d)\n", outLoopLimit, nTwiddle);
      final double[][] twid = ret[i];
      for (int j = 0; j < factor - 1; j++) {
        twid[0][2 * j] = 1.0;
        twid[0][2 * j + 1] = 0.0;
        // System.out.printf("    T(%d,%d) = %10.6f %10.6f\n", 0, j, twid[0][2 * j], twid[0][2 * j + 1]);
      }
      for (int k = 1; k < outLoopLimit; k++) {
        int m = 0;
        for (int j = 0; j < nTwiddle; j++) {
          m += k * product_1;
          m %= n;
          final double theta = TwoPI_N * m;
          twid[k][2 * j] = cos(theta);
          twid[k][2 * j + 1] = sin(theta);
          // System.out.printf("    T(%d,%d) = %10.6f %10.6f\n", k, j, twid[k][2 * j], twid[k][2 * j + 1]);
        }
      }
    }
    return ret;
  }

  /**
   * Static DFT method used to test the FFT.
   *
   * @param in  input array.
   * @param out output array.
   */
  public static void dft(double[] in, double[] out) {
    int n = in.length / 2;
    for (int k = 0; k < n; k++) { // For each output element
      double sumReal = 0;
      double simImag = 0;
      for (int t = 0; t < n; t++) { // For each input element
        double angle = (2 * PI * t * k) / n;
        int re = 2 * t;
        int im = 2 * t + 1;
        sumReal = fma(in[re], cos(angle), sumReal);
        sumReal = fma(in[im], sin(angle), sumReal);
        simImag = fma(-in[re], sin(angle), simImag);
        simImag = fma(in[im], cos(angle), simImag);
      }
      int re = 2 * k;
      int im = 2 * k + 1;
      out[re] = sumReal;
      out[im] = simImag;
    }
  }

  /**
   * Test the Complex FFT.
   *
   * @param args an array of {@link java.lang.String} objects.
   * @throws java.lang.Exception if any.
   * @since 1.0
   */
  public static void main(String[] args) throws Exception {
    int dimNotFinal = 128;
    int reps = 5;
    try {
      dimNotFinal = Integer.parseInt(args[0]);
      if (dimNotFinal < 1) {
        dimNotFinal = 100;
      }
      reps = Integer.parseInt(args[1]);
      if (reps < 1) {
        reps = 5;
      }
    } catch (Exception e) {
      //
    }
    final int dim = dimNotFinal;
    System.out.printf("Initializing a 1D array of length %d.\n"
        + "The best timing out of %d repetitions will be used.%n", dim, reps);
    Complex complex = new Complex(dim);
    final double[] data = new double[dim * 2];
    Random random = new Random(1);
    for (int i = 0; i < dim; i++) {
      data[2 * i] = random.nextDouble();
    }
    double toSeconds = 0.000000001;
    long seqTime = Long.MAX_VALUE;
    for (int i = 0; i < reps; i++) {
      System.out.printf("Iteration %d%n", i + 1);
      long time = System.nanoTime();
      complex.fft(data, 0, 2);
      complex.ifft(data, 0, 2);
      time = (System.nanoTime() - time);
      System.out.printf("Sequential: %12.9f%n", toSeconds * time);
      if (time < seqTime) {
        seqTime = time;
      }
    }
    System.out.printf("Best Sequential Time:  %12.9f%n", toSeconds * seqTime);
  }

}

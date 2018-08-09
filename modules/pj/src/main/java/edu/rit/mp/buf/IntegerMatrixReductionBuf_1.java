//******************************************************************************
//
// File:    IntegerMatrixReductionBuf_1.java
// Package: edu.rit.mp.buf
// Unit:    Class edu.rit.mp.buf.IntegerMatrixReductionBuf_1
//
// This Java source file is copyright (C) 2009 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java Library ("PJ"). PJ is free
// software; you can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// PJ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// Linking this library statically or dynamically with other modules is making a
// combined work based on this library. Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of this library give you
// permission to link this library with independent modules to produce an
// executable, regardless of the license terms of these independent modules, and
// to copy and distribute the resulting executable under terms of your choice,
// provided that you also meet, for each linked independent module, the terms
// and conditions of the license of that module. An independent module is a module
// which is not derived from or based on this library. If you modify this library,
// you may extend this exception to your version of the library, but you are not
// obligated to do so. If you do not wish to do so, delete this exception
// statement from your version.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************
package edu.rit.mp.buf;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import edu.rit.mp.Buf;
import edu.rit.mp.IntegerBuf;
import edu.rit.pj.reduction.IntegerOp;
import edu.rit.pj.reduction.Op;
import edu.rit.pj.reduction.ReduceArrays;
import edu.rit.util.Range;

/**
 * Class IntegerMatrixReductionBuf_1 provides a reduction buffer for class
 * {@linkplain IntegerMatrixBuf_1}.
 *
 * @author Alan Kaminsky
 * @version 05-Apr-2009
 */
class IntegerMatrixReductionBuf_1
        extends IntegerMatrixBuf_1 {

// Hidden data members.
    IntegerOp myOp;

// Exported constructors.
    /**
     * Construct a new integer matrix reduction buffer. It is assumed that the
     * rows and columns of <TT>theMatrix</TT> are allocated and that each row of
     * <TT>theMatrix</TT> has the same number of columns.
     *
     * @param theMatrix Matrix.
     * @param theRowRange Range of rows to include. The stride is assumed to be
     * 1.
     * @param theColRange Range of columns to include. The stride is assumed to
     * be 1.
     * @param op Binary operation.
     * @exception NullPointerException (unchecked exception) Thrown if
     * <TT>op</TT> is null.
     */
    public IntegerMatrixReductionBuf_1(int[][] theMatrix,
            Range theRowRange,
            Range theColRange,
            IntegerOp op) {
        super(theMatrix, theRowRange, theColRange);
        if (op == null) {
            throw new NullPointerException("IntegerMatrixReductionBuf_1(): op is null");
        }
        myOp = op;
    }

// Exported operations.
    /**
     * {@inheritDoc}
     *
     * Store the given item in this buffer.
     * <P>
     * The <TT>put()</TT> method must not block the calling thread; if it does,
     * all message I/O in MP will be blocked.
     */
    public void put(int i,
            int item) {
        int row = (i / myColCount) + myLowerRow;
        int col = (i % myColCount) + myLowerCol;
        myMatrix[row][col] = myOp.op(myMatrix[row][col], item);
    }

    /**
     * {@inheritDoc}
     *
     * Copy items from the given buffer to this buffer. The number of items
     * copied is this buffer's length or <TT>theSrc</TT>'s length, whichever is
     * smaller. If <TT>theSrc</TT> is this buffer, the <TT>copy()</TT> method
     * does nothing.
     * @exception ClassCastException (unchecked exception) Thrown if
     * <TT>theSrc</TT>'s item data type is not the same as this buffer's item
     * data type.
     */
    public void copy(Buf theSrc) {
        if (theSrc == this) {
        } else if (theSrc instanceof IntegerMatrixBuf) {
            IntegerMatrixBuf src = (IntegerMatrixBuf) theSrc;
            ReduceArrays.reduce(src.myMatrix, src.myRowRange, src.myColRange,
                    this.myMatrix, this.myRowRange, this.myColRange,
                    myOp);
        } else {
            IntegerBuf.defaultCopy((IntegerBuf) theSrc, this);
        }
    }

    /**
     * {@inheritDoc}
     *
     * Create a buffer for performing parallel reduction using the given binary
     * operation. The results of the reduction are placed into this buffer.
     * @exception ClassCastException (unchecked exception) Thrown if this
     * buffer's element data type and the given binary operation's argument data
     * type are not the same.
     */
    public Buf getReductionBuf(Op op) {
        throw new UnsupportedOperationException();
    }

// Hidden operations.
    /**
     * {@inheritDoc}
     *
     * Receive as many items as possible from the given byte buffer to this
     * buffer.
     * <P>
     * The <TT>receiveItems()</TT> method must not block the calling thread; if
     * it does, all message I/O in MP will be blocked.
     */
    protected int receiveItems(int i,
            int num,
            ByteBuffer buffer) {
        IntBuffer intbuffer = buffer.asIntBuffer();
        num = Math.min(num, intbuffer.remaining());
        int n = 0;
        int r = i2r(i);
        int row = r + myLowerRow;
        int c = i2c(i);
        int col = c + myLowerCol;
        int ncols = Math.min(myColCount - c, num);
        while (r < myRowCount && ncols > 0) {
            int[] myMatrix_row = myMatrix[row];
            for (c = 0; c < ncols; ++c) {
                myMatrix_row[col]
                        = myOp.op(myMatrix_row[col], intbuffer.get());
                ++col;
            }
            num -= ncols;
            n += ncols;
            ++r;
            ++row;
            col = myLowerCol;
            ncols = Math.min(myColCount, num);
        }
        buffer.position(buffer.position() + 4 * n);
        return n;
    }

}

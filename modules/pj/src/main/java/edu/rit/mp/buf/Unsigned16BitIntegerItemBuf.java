//******************************************************************************
//
// File:    Unsigned16BitIntegerItemBuf.java
// Package: edu.rit.mp.buf
// Unit:    Class edu.rit.mp.buf.Unsigned16BitIntegerItemBuf
//
// This Java source file is copyright (C) 2007 by Alan Kaminsky. All rights
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

import edu.rit.mp.Buf;
import edu.rit.mp.Unsigned16BitIntegerBuf;
import edu.rit.pj.reduction.IntegerOp;
import edu.rit.pj.reduction.Op;

/**
 * Class Unsigned16BitIntegerItemBuf provides a buffer for a single unsigned
 * 16-bit integer item sent or received using the Message Protocol (MP). While
 * an instance of class Unsigned16BitIntegerItemBuf may be constructed directly,
 * normally you will use a factory method in class {@linkplain
 * edu.rit.mp.Unsigned16BitIntegerBuf Unsigned16BitIntegerBuf}. See that class
 * for further information.
 *
 * @author Alan Kaminsky
 * @version 26-Oct-2007
 */
public class Unsigned16BitIntegerItemBuf
        extends Unsigned16BitIntegerBuf {

// Exported data members.
    /**
     * Integer item to be sent or received.
     */
    public int item;

// Exported constructors.
    /**
     * Construct a new unsigned 16-bit integer item buffer.
     */
    public Unsigned16BitIntegerItemBuf() {
        super(1);
    }

    /**
     * Construct a new unsigned 16-bit integer item buffer with the given
     * initial value.
     *
     * @param item Initial value of the {@link #item} field.
     */
    public Unsigned16BitIntegerItemBuf(int item) {
        super(1);
        this.item = item;
    }

// Exported operations.
    /**
     * {@inheritDoc}
     *
     * Obtain the given item from this buffer.
     * <P>
     * The <TT>get()</TT> method must not block the calling thread; if it does,
     * all message I/O in MP will be blocked.
     */
    public int get(int i) {
        return this.item;
    }

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
        this.item = item;
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
        return new Unsigned16BitIntegerItemReductionBuf(this, (IntegerOp) op);
    }

// Hidden operations.
    /**
     * {@inheritDoc}
     *
     * Send as many items as possible from this buffer to the given byte buffer.
     * <P>
     * The <TT>sendItems()</TT> method must not block the calling thread; if it
     * does, all message I/O in MP will be blocked.
     */
    protected int sendItems(int i,
            ByteBuffer buffer) {
        if (buffer.remaining() >= 2) {
            buffer.putShort((short) this.item);
            return 1;
        } else {
            return 0;
        }
    }

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
        if (num >= 1 && buffer.remaining() >= 2) {
            this.item = buffer.getShort() & 0xFFFF;
            return 1;
        } else {
            return 0;
        }
    }

}

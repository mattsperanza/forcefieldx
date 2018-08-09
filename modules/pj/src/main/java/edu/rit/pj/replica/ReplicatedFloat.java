//******************************************************************************
//
// File:    ReplicatedFloat.java
// Package: edu.rit.pj.replica
// Unit:    Class edu.rit.pj.replica.ReplicatedFloat
//
// This Java source file is copyright (C) 2008 by Alan Kaminsky. All rights
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
package edu.rit.pj.replica;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import edu.rit.mp.FloatBuf;
import edu.rit.mp.buf.FloatItemBuf;
import edu.rit.pj.Comm;
import edu.rit.pj.reduction.FloatOp;

/**
 * Class ReplicatedFloat provides a replicated, shared reduction variable for a
 * value of type <TT>float</TT>.
 * <P>
 * A replicated, shared reduction variable is intended to be used in a cluster
 * or hybrid parallel program for a data item shared among all the processes in
 * the program and all the threads in each process. To use a replicated, shared
 * reduction variable, do the following in each process of the parallel program:
 * <OL TYPE=1>
 * <LI>
 * Construct an instance of class ReplicatedFloat, specifying the reduction
 * operator (class {@linkplain edu.rit.pj.reduction.Op}) to use when performing
 * updates, and specifying the communicator (class {@linkplain edu.rit.pj.Comm})
 * and the message tag to use for sending updates among the processes. At this
 * point a <I>replica</I> of the variable exists in each process.
 * <P>
 * <LI>
 * To read the variable, call the <TT>get()</TT> method. The current value of
 * the local process's replicated variable is returned.
 * <P>
 * <LI>
 * To update the variable, call the <TT>reduce()</TT> method, specifying a new
 * value. The <TT>reduce()</TT> method performs an <I>atomic reduction</I>
 * (described below) on the local process's replicated variable with the new
 * value. If the variable changed as a result of the reduction, the variable's
 * (updated) value is flooded to all the processes in the communicator. Finally,
 * the <TT>reduce()</TT> method returns the variable's value.
 * <P>
 * Whenever one of the aforementioned flooded messages arrives, a separate
 * thread performs an atomic reduction on the local process's variable with the
 * received value.
 * <P>
 * An atomic reduction consists of these steps, performed atomically: Call the
 * reduction operator's <TT>op()</TT> method, passing in the current value of
 * the local process's replicated variable and the new value (either the new
 * value specified as an argument of <TT>reduce()</TT>, or the new value
 * received in a flooded message). Then store the <TT>op()</TT> method's return
 * value back into the local process's replicated variable.
 * </OL>
 * <P>
 * Class ReplicatedFloat does not itself guarantee consistency of the replicas'
 * values. This is to avoid the message passing overhead of a distributed state
 * update protocol. Instead, the parallel program must be written to operate
 * correctly when the variable is updated as described above. Note that the
 * value of a process's local replica can change asynchronously at any time,
 * either because a thread in the current process updated the variable, or
 * because a flooded message updated the variable.
 * <P>
 * Class ReplicatedFloat is multiple thread safe. The methods use lock-free
 * atomic compare-and-set.
 * <P>
 * <I>Note:</I> Class ReplicatedFloat is implemented using class
 * java.util.concurrent.atomic.AtomicInteger.
 *
 * @author Alan Kaminsky
 * @version 13-Sep-2008
 */
public class ReplicatedFloat
        extends Number {

// Hidden data members.
    private FloatOp myOp;
    private AtomicInteger myValue;
    private int myTag;
    private Comm myComm;
    private Receiver myReceiver;

// Hidden helper classes.
    /**
     * Class Receiver receives and processes flooded messages to update this
     * replicated, shared reduction variable.
     *
     * @author Alan Kaminsky
     * @version 13-Sep-2008
     */
    private class Receiver
            extends Thread {

        public void run() {
            FloatItemBuf buf = FloatBuf.buffer();

            try {
                for (;;) {
                    // Receive a flooded message.
                    myComm.floodReceive(myTag, buf);

                    // Do an atomic reduction.
                    int oldint, newint;
                    float oldvalue, newvalue;
                    do {
                        oldint = myValue.get();
                        oldvalue = Float.intBitsToFloat(oldint);
                        newvalue = myOp.op(oldvalue, buf.item);
                        newint = Float.floatToIntBits(newvalue);
                    } while (!myValue.compareAndSet(oldint, newint));
                }
            } catch (Throwable exc) {
                exc.printStackTrace(System.err);
            }
        }
    }

// Exported constructors.
    /**
     * Construct a new replicated, shared float reduction variable with the
     * given reduction operator. The initial value is 0. A message tag of 0 is
     * used. The world communicator is used.
     *
     * @param op Reduction operator.
     * @exception NullPointerException (unchecked exception) Thrown if
     * <TT>op</TT> is null.
     */
    public ReplicatedFloat(FloatOp op) {
        this(op, 0.0f, 0, Comm.world());
    }

    /**
     * Construct a new replicated, shared float reduction variable with the
     * given reduction operator and initial value. A message tag of 0 is used.
     * The world communicator is used.
     *
     * @param op Reduction operator.
     * @param initialValue Initial value.
     * @exception NullPointerException (unchecked exception) Thrown if
     * <TT>op</TT> is null.
     */
    public ReplicatedFloat(FloatOp op,
            float initialValue) {
        this(op, initialValue, 0, Comm.world());
    }

    /**
     * Construct a new replicated, shared float reduction variable with the
     * given reduction operator, initial value, and message tag. The world
     * communicator is used.
     *
     * @param op Reduction operator.
     * @param initialValue Initial value.
     * @param tag Message tag.
     * @exception NullPointerException (unchecked exception) Thrown if
     * <TT>op</TT> is null. Thrown if
     * <TT>comm</TT> is null.
     */
    public ReplicatedFloat(FloatOp op,
            float initialValue,
            int tag) {
        this(op, initialValue, tag, Comm.world());
    }

    /**
     * Construct a new replicated, shared float reduction variable with the
     * given reduction operator, initial value, message tag, and communicator.
     *
     * @param op Reduction operator.
     * @param initialValue Initial value.
     * @param tag Message tag.
     * @param comm Communicator.
     * @exception NullPointerException (unchecked exception) Thrown if
     * <TT>op</TT> is null. Thrown if
     * <TT>comm</TT> is null.
     */
    public ReplicatedFloat(FloatOp op,
            float initialValue,
            int tag,
            Comm comm) {
        if (op == null) {
            throw new NullPointerException("ReplicatedFloat(): op is null");
        }
        if (comm == null) {
            throw new NullPointerException("ReplicatedFloat(): comm is null");
        }
        myOp = op;
        myValue = new AtomicInteger(Float.floatToIntBits(initialValue));
        myTag = tag;
        myComm = comm;
        myReceiver = new Receiver();
        myReceiver.setDaemon(true);
        myReceiver.start();
    }

// Exported operations.
    /**
     * Returns this replicated, shared reduction variable's current value.
     *
     * @return Current value.
     */
    public float get() {
        return Float.intBitsToFloat(myValue.get());
    }

    /**
     * Update this replicated, shared reduction variable's current value. This
     * variable is combined with the given value using the reduction operation
     * specified to the constructor (<I>op</I>). The result is stored back into
     * this variable and is returned; the result may also be flooded to all
     * processes in the communicator.
     *
     * @param value Value.
     * @return (This variable) <I>op</I> (<TT>value</TT>).
     * @exception IOException Thrown if an I/O error occurred.
     * @throws java.io.IOException if any.
     */
    public float reduce(float value)
            throws IOException {
        // Do an atomic reduction.
        int oldint, newint;
        float oldvalue, newvalue;
        do {
            oldint = myValue.get();
            oldvalue = Float.intBitsToFloat(oldint);
            newvalue = myOp.op(oldvalue, value);
            newint = Float.floatToIntBits(newvalue);
        } while (!myValue.compareAndSet(oldint, newint));

        // If value changed, send a flooded message.
        if (newvalue != oldvalue) {
            myComm.floodSend(myTag, FloatBuf.buffer(newvalue));
        }

        // Return updated value.
        return newvalue;
    }

    /**
     * Returns a string version of this reduction variable.
     *
     * @return String version.
     */
    public String toString() {
        return Float.toString(get());
    }

    /**
     * Returns this reduction variable's current value converted to type
     * <TT>int</TT>.
     *
     * @return Current value.
     */
    public int intValue() {
        return (int) get();
    }

    /**
     * Returns this reduction variable's current value converted to type
     * <TT>long</TT>.
     *
     * @return Current value.
     */
    public long longValue() {
        return (long) get();
    }

    /**
     * Returns this reduction variable's current value converted to type
     * <TT>float</TT>.
     *
     * @return Current value.
     */
    public float floatValue() {
        return (float) get();
    }

    /**
     * Returns this reduction variable's current value converted to type
     * <TT>double</TT>.
     *
     * @return Current value.
     */
    public double doubleValue() {
        return (double) get();
    }

}

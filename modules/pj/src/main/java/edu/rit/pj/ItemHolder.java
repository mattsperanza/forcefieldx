//******************************************************************************
//
// File:    ItemHolder.java
// Package: edu.rit.pj
// Unit:    Class edu.rit.pj.ItemHolder
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
package edu.rit.pj;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serial;

/**
 * Class ItemHolder provides an object that holds one item to be processed by a
 * {@linkplain ParallelIteration} along with associated information.
 *
 * @param <T> Data type of the items iterated over.
 *
 * @author Alan Kaminsky
 * @version 07-Oct-2010
 */
@SuppressWarnings("serial")
class ItemHolder<T>
        implements Externalizable {

// Hidden data members.
    @Serial
    private static final long serialVersionUID = -5475933248018589590L;

// Exported data members.
    /**
     * The item itself (may be null).
     */
    public T myItem;

    /**
     * The item's sequence number in the iteration. Sequence numbers start at 0
     * and increase by 1 for each item.
     */
    public int mySequenceNumber;

// Exported constructors.
    /**
     * Construct a new item holder.
     */
    public ItemHolder() {
    }

// Exported operations.
    /**
     * {@inheritDoc}
     *
     * Write this object holder to the given object output stream.
     * @exception IOException Thrown if an I/O error occurred.
     */
    public void writeExternal(ObjectOutput out)
            throws IOException {
        out.writeObject(myItem);
        out.writeInt(mySequenceNumber);
    }

    /**
     * {@inheritDoc}
     *
     * Read this object holder from the given object input stream.
     * @exception IOException Thrown if an I/O error occurred.
     * @exception ClassNotFoundException Thrown if the class of the object could
     * not be found.
     */
    @SuppressWarnings("unchecked")
    public void readExternal(ObjectInput in)
            throws IOException, ClassNotFoundException {
        myItem = (T) in.readObject();
        mySequenceNumber = in.readInt();
    }

}

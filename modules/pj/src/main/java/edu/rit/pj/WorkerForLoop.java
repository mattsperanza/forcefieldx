//******************************************************************************
//
// File:    WorkerForLoop.java
// Package: edu.rit.pj
// Unit:    Class edu.rit.pj.WorkerForLoop
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
package edu.rit.pj;

/**
 * Class WorkerForLoop is the abstract base class for a worker for loop that is
 * executed inside a {@linkplain WorkerRegion}. There are four variations of a
 * worker for loop, depending on the loop index data type and on whether the
 * loop stride is implicit or explicit. There is a subclass for each variation;
 * see the subclasses for further information. The subclasses are:
 * <UL>
 * <P>
 * <LI>
 * Class {@linkplain WorkerIntegerForLoop} -- loop index is type <TT>int</TT>,
 * loop stride is implicit (+1).
 * <P>
 * <LI>
 * Class {@linkplain WorkerIntegerStrideForLoop} -- loop index is type
 * <TT>int</TT>, loop stride is explicitly specified.
 * <P>
 * <LI>
 * Class {@linkplain WorkerLongForLoop} -- loop index is type <TT>long</TT>,
 * loop stride is implicit (+1).
 * <P>
 * <LI>
 * Class {@linkplain WorkerLongStrideForLoop} -- loop index is type
 * <TT>long</TT>, loop stride is explicitly specified.
 * </UL>
 *
 * @author Alan Kaminsky
 * @version 17-Nov-2009
 */
public abstract class WorkerForLoop
        extends WorkerConstruct {

// Exported constructors.
    /**
     * Construct a new worker for loop.
     */
    public WorkerForLoop() {
        super();
    }

}

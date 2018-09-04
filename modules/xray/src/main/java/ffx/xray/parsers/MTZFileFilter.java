/**
 * Title: Force Field X.
 * <p>
 * Description: Force Field X - Software for Molecular Biophysics.
 * <p>
 * Copyright: Copyright (c) Michael J. Schnieders 2001-2018.
 * <p>
 * This file is part of Force Field X.
 * <p>
 * Force Field X is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as published by
 * the Free Software Foundation.
 * <p>
 * Force Field X is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * Force Field X; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 * <p>
 * Linking this library statically or dynamically with other modules is making a
 * combined work based on this library. Thus, the terms and conditions of the
 * GNU General Public License cover the whole combination.
 * <p>
 * As a special exception, the copyright holders of this library give you
 * permission to link this library with independent modules to produce an
 * executable, regardless of the license terms of these independent modules, and
 * to copy and distribute the resulting executable under terms of your choice,
 * provided that you also meet, for each linked independent module, the terms
 * and conditions of the license of that module. An independent module is a
 * module which is not derived from or based on this library. If you modify this
 * library, you may extend this exception to your version of the library, but
 * you are not obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
 */
package ffx.xray.parsers;

import javax.swing.filechooser.FileFilter;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * The MTZFileFilter class is used to choose CCP4 MTZ files
 *
 * @author Michael J. Schnieders
 *
 * @since 1.0
 */
public final class MTZFileFilter extends FileFilter {

    /**
     * Default Constructor.
     */
    public MTZFileFilter() {
    }

    /**
     * {@inheritDoc}
     *
     * This method determines whether or not the file parameter is an *.mtz file
     * or not, returning true if it is (true is also returned for any directory)
     */
    @Override
    public boolean accept(File file) {
        if (file.isDirectory()) {
            return true;
        }
        String fileName = file.getName().toLowerCase();
        return fileName.endsWith(".mtz");
    }

    /**
     * <p>
     * acceptDeep</p>
     *
     * @param file a {@link java.io.File} object.
     * @return a boolean.
     */
    public boolean acceptDeep(File file) {
        try {
            if (file == null || file.isDirectory() || !file.canRead()) {
                return false;
            }
            FileInputStream fileInputStream = new FileInputStream(file);
            DataInputStream dataInputStream = new DataInputStream(fileInputStream);

            byte bytes[] = new byte[80];
            int offset = 0;

            // Is this an MTZ file?
            dataInputStream.read(bytes, offset, 4);
            String mtzstr = new String(bytes);
            if (!mtzstr.trim().equals("MTZ")) {
                return false;
            }

            return true;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * {@inheritDoc}
     *
     * Provides a description of this FileFilter.
     */
    @Override
    public String getDescription() {
        return new String("CCP4 MTZ Reflection Files: *.mtz");
    }
}

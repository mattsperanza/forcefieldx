/**
 * Title: Force Field X
 * Description: Force Field X - Software for Molecular Biophysics.
 * Copyright: Copyright (c) Michael J. Schnieders 2001-2009
 *
 * This file is part of Force Field X.
 *
 * Force Field X is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as published
 * by the Free Software Foundation.
 *
 * Force Field X is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Force Field X; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
 */
package ffx.potential.parameters;

import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NavigableSet;
import java.util.TreeMap;
import java.util.logging.Logger;

import org.apache.commons.configuration.CompositeConfiguration;

/**
 * The ForceField class organizes parameters for a molecular mechanics force
 * field.
 *
 * @author Michael J. Schnieders
 *
 * @since 1.0
 */
public class ForceField {

    private static final Logger logger = Logger.getLogger(ForceField.class.getName());

    /**
     * Available force fields; currently limited to the AMOEBA family.
     */
    public enum Force_Field {

        AMOEBA_WATER, AMOEBA_2004, AMOEBA_PROTEIN_2004,
        AMOEBA_2009, AMOEBA_PROTEIN_2009, AMOEBA_NUCLEIC_2009,
        AMOEBA_BIO_2009
    }

    public enum ForceFieldString {

        SPACEGROUP, FORCEFIELD, POLARIZATION
    }

    public enum ForceFieldDouble {

        A_AXIS, B_AXIS, C_AXIS, ALPHA, BETA, GAMMA, POLAR_DAMP, POLAR_SOR,
        POLAR_EPS, EWALD_CUTOFF, EWALD_ALPHA, PME_SPACING, VDW_CUTOFF,
        MPOLE_11_SCALE, MPOLE_12_SCALE, MPOLE_13_SCALE, MPOLE_14_SCALE,
        MPOLE_15_SCALE, POLAR_12_SCALE, POLAR_13_SCALE, DIRECT_11_SCALE
    }

    public enum ForceFieldInteger {

        PME_ORDER
    }

    public enum ForceFieldBoolean {

        BONDTERM, ANGLETERM, STRBNDTERM, UREYTERM, OPBENDTERM,
        TORSIONTERM, PITORSTERM, TORTORTERM, VDWTERM,
        MPOLETERM, POLARIZETERM, SCFCACHE, OPENCL;
    }

    public enum ForceFieldType {

        KEYWORD, ANGLE, ATOM, BIOTYPE, BOND, CHARGE, MULTIPOLE, OPBEND, PITORS, POLARIZE,
        STRBND, TORSION, TORTORS, UREYBRAD, VDW
    }
    /**
     * A map between a Force_Field and its internal parameter file.
     */
    private static final TreeMap<Force_Field, URL> forceFields = new TreeMap<Force_Field, URL>();

    static {
        ClassLoader cl = ForceField.class.getClassLoader();
        String prefix = "ffx/potential/parameters/amoeba/";
        Force_Field ff = Force_Field.AMOEBA_WATER;
        forceFields.put(ff, cl.getResource(prefix + ff));
        ff = Force_Field.AMOEBA_2004;
        forceFields.put(ff, cl.getResource(prefix + ff));
        ff = Force_Field.AMOEBA_PROTEIN_2004;
        forceFields.put(ff, cl.getResource(prefix + ff));
        ff = Force_Field.AMOEBA_2009;
        forceFields.put(ff, cl.getResource(prefix + ff));
        ff = Force_Field.AMOEBA_PROTEIN_2009;
        String value = prefix + ff;
        forceFields.put(ff, cl.getResource(value));
        ff = Force_Field.AMOEBA_NUCLEIC_2009;
        forceFields.put(ff, cl.getResource(value));
        ff = Force_Field.AMOEBA_BIO_2009;
        forceFields.put(ff, cl.getResource(prefix + ff));
    }
    public URL forceFieldURL;
    public File keywordFile;
    private final CompositeConfiguration properties;
    private final TreeMap<String, AngleType> angleTypes;
    private final TreeMap<String, AtomType> atomTypes;
    private final TreeMap<String, BioType> bioTypes;
    private final TreeMap<String, BondType> bondTypes;
    private final TreeMap<String, MultipoleType> multipoleTypes;
    private final TreeMap<String, OutOfPlaneBendType> outOfPlaneBendTypes;
    private final TreeMap<String, PolarizeType> polarizeTypes;
    private final TreeMap<String, StretchBendType> stretchBendTypes;
    private final TreeMap<String, PiTorsionType> piTorsionTypes;
    private final TreeMap<String, TorsionType> torsionTypes;
    private final TreeMap<String, TorsionTorsionType> torsionTorsionTypes;
    private final TreeMap<String, UreyBradleyType> ureyBradleyTypes;
    private final TreeMap<String, VDWType> vanderWaalsTypes;
    private final TreeMap<ForceFieldType, TreeMap> forceFieldTypes;

    public static URL getForceFieldURL(Force_Field forceField) {
        if (forceField != null) {
            return forceFields.get(forceField);
        } else {
            return null;
        }
    }

    /**
     * ForceField Constructor.
     */
    public ForceField(CompositeConfiguration properties, File parameterFile) {
        this(properties);
        this.keywordFile = parameterFile;
    }

    /**
     * ForceField Constructor.
     */
    public ForceField(CompositeConfiguration properties) {
        this.properties = properties;
        /**
         * Each force field "type" implements the "Comparator<String>" interface
         * so that passing an "empty" instance of the "type" to its TreeMap
         * constructor will keep the types sorted.
         */
        angleTypes = new TreeMap<String, AngleType>(new AngleType(new int[3], 0, new double[1]));
        atomTypes = new TreeMap<String, AtomType>(new AtomType(0, 0, null, null, 0, 0, 0));
        bioTypes = new TreeMap<String, BioType>(new BioType(0, null, null, 0));
        bondTypes = new TreeMap<String, BondType>(new BondType(new int[2], 0, 0));
        multipoleTypes = new TreeMap<String, MultipoleType>(new MultipoleType(0, new double[3], new double[3][3], null, null));
        outOfPlaneBendTypes = new TreeMap<String, OutOfPlaneBendType>(new OutOfPlaneBendType(new int[4], 0));
        piTorsionTypes = new TreeMap<String, PiTorsionType>(new PiTorsionType(new int[2], 0));
        polarizeTypes = new TreeMap<String, PolarizeType>(new PolarizeType(0, 0, 0, new int[1]));
        stretchBendTypes = new TreeMap<String, StretchBendType>(new StretchBendType(new int[3], new double[1]));
        torsionTorsionTypes = new TreeMap<String, TorsionTorsionType>();
        torsionTypes = new TreeMap<String, TorsionType>(new TorsionType(new int[4], new double[1], new double[1], new int[1]));
        ureyBradleyTypes = new TreeMap<String, UreyBradleyType>(new UreyBradleyType(new int[3], 0, 0));
        vanderWaalsTypes = new TreeMap<String, VDWType>(new VDWType(0, 0, 0, 0));

        forceFieldTypes = new TreeMap<ForceFieldType, TreeMap>();
        forceFieldTypes.put(ForceFieldType.ANGLE, angleTypes);
        forceFieldTypes.put(ForceFieldType.ATOM, atomTypes);
        forceFieldTypes.put(ForceFieldType.BOND, bondTypes);
        forceFieldTypes.put(ForceFieldType.BIOTYPE, bioTypes);
        forceFieldTypes.put(ForceFieldType.OPBEND, outOfPlaneBendTypes);
        forceFieldTypes.put(ForceFieldType.MULTIPOLE, multipoleTypes);
        forceFieldTypes.put(ForceFieldType.PITORS, piTorsionTypes);
        forceFieldTypes.put(ForceFieldType.POLARIZE, polarizeTypes);
        forceFieldTypes.put(ForceFieldType.STRBND, stretchBendTypes);
        forceFieldTypes.put(ForceFieldType.TORSION, torsionTypes);
        forceFieldTypes.put(ForceFieldType.TORTORS, torsionTorsionTypes);
        forceFieldTypes.put(ForceFieldType.UREYBRAD, ureyBradleyTypes);
        forceFieldTypes.put(ForceFieldType.VDW, vanderWaalsTypes);
    }

    /**
     * Enums are uppercase with underscores, but property files use lower case
     * with dashes.
     *
     * @param s
     * @return
     */
    private String normalizeKey(String s) {
        if (s == null) return null;
        return s.toLowerCase().replace("_", "-");
    }

    public double getDouble(ForceFieldDouble forceFieldDouble)
            throws Exception {
        if (forceFieldDouble == null) {
            throw new Exception("NULL keyword");
        }
        String key = normalizeKey(forceFieldDouble.toString());
        if (!properties.containsKey(key)) {
            throw new Exception("Undefined keyword: " + key);
        }
        return properties.getDouble(key);
    }

    public double getDouble(ForceFieldDouble forceFieldDouble,
            double defaultValue) {
        try {
            return getDouble(forceFieldDouble);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    public int getInteger(ForceFieldInteger forceFieldInteger)
            throws Exception {
        if (forceFieldInteger == null) {
            throw new Exception("NULL keyword");
        }
        String key = normalizeKey(forceFieldInteger.toString());
        if (!properties.containsKey(key)) {
            throw new Exception("Undefined keyword: " + key);
        }
        return properties.getInt(key);
    }

    public int getInteger(ForceFieldInteger forceFieldInteger,
            int defaultValue) {
        try {
            return getInteger(forceFieldInteger);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    public String getString(ForceFieldString forceFieldString)
            throws Exception {
        if (forceFieldString == null) {
            throw new Exception("NULL keyword");
        }
        String key = normalizeKey(forceFieldString.toString());
        if (!properties.containsKey(key)) {
            throw new Exception("Undefined keyword: " + key);
        }
        return properties.getString(key);
    }

    public String getString(ForceFieldString forceFieldString,
            String defaultString) {
        try {
            return getString(forceFieldString);
        } catch (Exception e) {
            return defaultString;
        }
    }

    public boolean getBoolean(ForceFieldBoolean forceFieldBoolean)
            throws Exception {
        if (forceFieldBoolean == null) {
            throw new Exception("NULL keyword");
        }
        String key = normalizeKey(forceFieldBoolean.toString());
        if (!properties.containsKey(key)) {
            throw new Exception("Undefined keyword: " + key);
        }
        return properties.getBoolean(key);
    }

    public boolean getBoolean(ForceFieldBoolean forceFieldBoolean,
            Boolean defaultBoolean) {
        try {
            return getBoolean(forceFieldBoolean);
        } catch (Exception e) {
            return defaultBoolean;
        }
    }

    /**
     * Add a force field keyword that is represented by a double.
     *
     * @param forceFieldDouble
     *            ForceFieldDouble
     * @param value
     *            double
     */
    public void addForceFieldDouble(ForceFieldDouble forceFieldDouble, double value) {
        if (forceFieldDouble == null) {
            return;
        }
        String key = normalizeKey(forceFieldDouble.toString());
        try {
            double old = getDouble(forceFieldDouble);
            logger.info("Clearing " + key + " with value " + old);
            properties.clearProperty(key);
        } catch (Exception e) {
            // Property does not exist yet.
        } finally {
            logger.info("Adding " + key + " with value " + value);
            properties.addProperty(key, value);
        }
    }

    /**
     * Add a force field keyword that is represented by an int.
     *
     * @param forceFieldInteger
     *            ForceFieldInteger
     * @param value
     *            int
     */
    public void addForceFieldInteger(ForceFieldInteger forceFieldInteger, int value) {
        if (forceFieldInteger == null) {
            return;
        }
        String key = normalizeKey(forceFieldInteger.toString());
        try {
            int old = getInteger(forceFieldInteger);
            logger.info("Clearing " + key + " with value " + old);
            properties.clearProperty(key);
        } catch (Exception e) {
            // Property does not exist yet.
        } finally {
            logger.info("Adding " + key + " with value " + value);
            properties.addProperty(key, value);
        }
    }

    /**
     * Store a force field keyword that is represented by a String.
     *
     * @param forceFieldString
     *            ForceFieldString
     * @param value
     *            String
     */
    public void addForceFieldString(ForceFieldString forceFieldString, String value) {
        if (forceFieldString == null) {
            return;
        }
        String key = normalizeKey(forceFieldString.toString());
        try {
            String old = getString(forceFieldString);
            logger.info("Clearing " + key + " with value " + old);
            properties.clearProperty(key);
        } catch (Exception e) {
            // Property does not exist yet.
        } finally {
            logger.info("Adding " + key + " with value " + value);
            properties.addProperty(key, value);
        }
    }

    /**
     * Store a force field keyword that is represented by a Boolean.
     *
     * @param forceFieldBoolean ForceFielBoolean
     *
     * @param value Boolean
     */
    public void addForceFieldBoolean(ForceFieldBoolean forceFieldBoolean, Boolean value) {
        if (forceFieldBoolean == null) {
            return;
        }
        String key = normalizeKey(forceFieldBoolean.toString());
        try {
            boolean old = getBoolean(forceFieldBoolean);
            logger.info("Clearing " + key + " with value " + old);
            properties.clearProperty(key);
        } catch (Exception e) {
            // Property does not exist yet.
        } finally {
            logger.info("Adding " + key + " with value " + value);
            properties.addProperty(key, value);
        }
    }

    /**
     * Add an instance of a force field type. Force Field types are more
     * complicated than simple Strings or doubles, in that they have multiple
     * fields and may occur multiple times.
     *
     * @param type
     *            BaseType
     */
    public void addForceFieldType(BaseType type) {
        if (type == null) {
            logger.info("Null force field type ignored.");
            return;
        }
        TreeMap treeMap = forceFieldTypes.get(type.forceFieldType);
        if (treeMap == null) {
            logger.info("Unrecognized force field type ignored " + type.forceFieldType);
            type.print();
        }
        if (treeMap.containsKey(type.key)) {
            logger.fine("A force field entry of type " + type.forceFieldType
                    + " already exists with the key: " + type.key
                    + "\nThe (discarded) old entry:\n" + treeMap.get(type.key).
                    toString() + "\nThe new entry:\n" + type.toString());
        }
        Class baseTypeClass = type.getClass();
        treeMap.put(type.key, baseTypeClass.cast(type));
    }

    public AngleType getAngleType(String key) {
        return angleTypes.get(key);
    }

    public AtomType getAtomType(String key) {
        return atomTypes.get(key);
    }

    public BondType getBondType(String key) {
        return bondTypes.get(key);
    }

    public BioType getBioType(String key) {
        return bioTypes.get(key);
    }

    /*
    public ChargeType getChargeType(String key) {
    return chargeTypes.get(key);
    } */
    public MultipoleType getMultipoleType(String key) {
        return multipoleTypes.get(key);
    }

    public OutOfPlaneBendType getOutOfPlaneBendType(String key) {
        return outOfPlaneBendTypes.get(key);
    }

    public PolarizeType getPolarizeType(String key) {
        return polarizeTypes.get(key);
    }

    public StretchBendType getStretchBendType(String key) {
        return stretchBendTypes.get(key);
    }

    public PiTorsionType getPiTorsionType(String key) {
        return piTorsionTypes.get(key);
    }

    public TorsionType getTorsionType(String key) {
        return torsionTypes.get(key);
    }

    public TorsionTorsionType getTorsionTorsionType(String key) {
        return torsionTorsionTypes.get(key);
    }

    public UreyBradleyType getUreyBradleyType(String key) {
        return ureyBradleyTypes.get(key);
    }

    public VDWType getVDWType(String key) {
        return vanderWaalsTypes.get(key);
    }

    public TreeMap<String, VDWType> getVDWTypes() {
        return vanderWaalsTypes;
    }

    public int getForceFieldTypeCount(ForceFieldType type) {
        TreeMap<String, BaseType> treeMap =
                (TreeMap<String, BaseType>) forceFieldTypes.get(type);
        if (treeMap == null) {
            logger.warning("Unrecognized Force Field Type: " + type);
            return 0;
        }
        return treeMap.size();
    }

    /**
     * Check for self-consistent polarization groups.
     */
    public void checkPolarizationTypes() {
        boolean change = false;
        for (String key : polarizeTypes.keySet()) {
            PolarizeType polarizeType = polarizeTypes.get(key);
            int orig = Integer.parseInt(key);
            int types[] = polarizeType.polarizationGroup;
            if (types == null) {
                continue;
            }
            for (int i = 0; i < types.length; i++) {
                int type = types[i];
                String key2 = Integer.toString(type);
                PolarizeType polarizeType2 = polarizeTypes.get(key2);
                if (polarizeType2 == null) {
                    logger.severe("Polarize type " + key
                            + "references nonexistant polarize type " + key2);
                    continue;
                }
                int types2[] = polarizeType2.polarizationGroup;
                if (types2 == null) {
                    polarizeType2.add(orig);
                    change = true;
                    continue;
                }
                boolean found = false;
                for (int j = 0; j < types2.length; j++) {
                    int type2 = types2[j];
                    for (int k = 0; k < types.length; k++) {
                        if (type2 == types[k]) {
                            found = true;
                        }
                    }
                    if (!found) {
                        polarizeType.add(type2);
                        change = true;
                    }
                }
            }
        }
        if (change) {
            checkPolarizationTypes();
        }
    }

    public void log() {
        for (ForceFieldType s : forceFieldTypes.keySet()) {
            log(s.toString());
        }
    }

    /**
     * Prints any force field keyword to Standard.out.
     *
     * @param key
     *            String
     */
    public void log(String key) {
        ForceFieldType type = ForceFieldType.valueOf(key);
        logger.info(toString(type));
    }

    public void print() {
        for (ForceFieldType s : forceFieldTypes.keySet()) {
            print(s.toString());
        }
    }

    public void print(String key) {
        ForceFieldType type = ForceFieldType.valueOf(key);
        System.out.println(toString(type));
    }

    /**
     * Return a String for any Force Field keyword.
     *
     * @param type
     *            ForceFieldType
     * @return String
     */
    public String toString(ForceFieldType type) {
        StringBuffer stringBuffer = new StringBuffer("\n");
        TreeMap t = forceFieldTypes.get(type);
        if (t.size() > 0) {
            NavigableSet set = t.navigableKeySet();
            Iterator i = set.iterator();
            while (i.hasNext()) {
                stringBuffer.append(t.get(i.next()) + "\n");
            }
            return stringBuffer.toString();
        }
        return "";
    }

    public String toString(String key) {
        if (key == null) {
            return null;
        }
        
        key = normalizeKey(key);

        if (properties.containsKey(key)) {
            List l = properties.getList(key);
            return key + " " + Arrays.toString(l.toArray());
        } else {
            return key + " is not defined.";
        }
    }
}

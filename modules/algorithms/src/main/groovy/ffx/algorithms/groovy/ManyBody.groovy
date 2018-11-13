package ffx.algorithms.groovy

import edu.rit.pj.Comm

import ffx.algorithms.RotamerOptimization
import ffx.algorithms.cli.AlgorithmsScript
import ffx.algorithms.cli.ManyBodyOptions
import ffx.numerics.Potential
import ffx.potential.ForceFieldEnergy
import ffx.potential.MolecularAssembly
import ffx.potential.bonded.Atom
import ffx.potential.bonded.Polymer
import ffx.potential.bonded.Residue
import ffx.potential.bonded.ResidueEnumerations
import ffx.potential.bonded.ResidueState
import ffx.potential.bonded.Rotamer
import ffx.potential.bonded.RotamerLibrary

import picocli.CommandLine.Command
import picocli.CommandLine.Mixin
import picocli.CommandLine.Parameters

import java.lang.reflect.Array

/**
 * The ManyBody script performs a discrete optimization using a many-body expansion and elimination expressions.
 * <br>
 * Usage:
 * <br>
 * ffxc ManyBody [options] &lt;filename&gt;
 */
@Command(description = " Run ManyBody algorithm on a system.", name = "ffxc ManyBody")
class ManyBody extends AlgorithmsScript {

    @Mixin
    ManyBodyOptions manyBody;

    /**
     * One or more filenames.
     */
    @Parameters(arity = "1..*", paramLabel = "files", description = "PDB input file.")
    private List<String> filenames;

    private File baseDir = null;
    boolean testing = null;

    ForceFieldEnergy potentialEnergy;

    boolean monteCarloTesting = false;

    @Override
    ManyBody run() {

        if (!init()) {
            return this;
        }

        String priorGKwarn = System.getProperty("gk-suppressWarnings");
        if (priorGKwarn == null || priorGKwarn.isEmpty()) {
            System.setProperty("gk-suppressWarnings", "true");
        }
        String modelFileName;
        if (filenames != null && filenames.size() > 0) {
            MolecularAssembly[] assemblies = algorithmFunctions.open(filenames.get(0));
            activeAssembly = assemblies[0];
            modelFileName = activeAssembly.getFile().getAbsolutePath();
        } else if (activeAssembly == null) {
            logger.info(helpString());
            return this;
        } else {
            // TODO: Get the active assembly from the User Interface/GUI. Or die.
            logger.warning("Could not load file or active assembly.");
        }
        activeAssembly.getPotentialEnergy().setPrintOnFailure(false, false);
        potentialEnergy = activeAssembly.getPotentialEnergy();

        // TODO: Check if a "rot" file exists; if so read it in and assign save rotamers to their respective residue.
        // Make sure file exists and is read in correctly
        int lenOfFile = modelFileName.length()
        int stop = lenOfFile-4

        String rotFileName = modelFileName.substring(0,stop)
        rotFileName = rotFileName.concat(".rot")

        // Try to open file
        File rotFile = new File(rotFileName)
        if(rotFile.canRead()){
            // Process file and assign useful variables (chainName/segID, resID/resNum, rotamer/rotamerName)
            println("Successfully read in rotamer file: "+rotFileName)
            String line
            ArrayList<String[]> rotFileContents = new ArrayList<>();
            try {
                rotFile.withReader { reader ->
                    while ((line = reader.readLine()) != null) {
                        logger.info("Inside line reader while loop")
                        // Split lines based on separator dictated in CreateRotamers
                        String[] lineArr = line.split(':')
                        logger.info("")

                        // Read into an Array List, for now
                        rotFileContents.add(lineArr)
                    }
                }
            } catch(Exception e){
                logger.warning("Exception caught: "+e.toString()+"\n")
                e.printStackTrace()
            } // End reading rot file reader try-catch block
            // TODO: Process the rotFileContents ArrayList to assign rotamers
            int rotFileContentsCounter = 0;
            while(rotFileContentsCounter < rotFileContents.size()){
                // Get a line
                String[] currentLine = rotFileContents.get(rotFileContentsCounter)
                /*for(int i = 0; i < currentLine.length; i++){
                    System.out.print(currentLine[i]+","+i+"\n")
                }*/

                // Assign Properties
                Character chainID = currentLine[3].toCharacter();
                String chainName = currentLine[4]
                int resID = currentLine[6].toInteger()
                String resName = currentLine[5]
                int rotNum = currentLine[9].toInteger()
                Polymer polymer = activeAssembly.getChain(chainName)
                Residue residue = polymer.getResidue(resID)

                String currentResName = resName
                int currentRotNum = rotNum

                // While the residue name remains the same, iterate through the associated rotamers
                while(currentResName.equals(resName)){
                    ArrayList<String[]> atomsInRotamer = new ArrayList<>();
                    System.out.println("Residue: "+currentResName)
                    // While the rotamer number remains the same, iterate through the associated atoms
                    while(rotNum == currentRotNum){
                        System.out.println("Rotamer Number: "+currentRotNum)
                        String atomName = currentLine[10]
                        String x = currentLine[12]
                        String y = currentLine[13]
                        String z = currentLine[14]
                        String[] singleAtomCoords = [atomName, x, y, z]
                        atomsInRotamer.add(singleAtomCoords)

                        // Update line counter
                        rotFileContentsCounter++
                        // Set new 'current' variables
                        if(rotFileContentsCounter < rotFileContents.size()) {
                            currentLine = rotFileContents.get(rotFileContentsCounter)
                            currentResName = currentLine[5]
                            currentRotNum = currentLine[9].toInteger()
                        } else{
                            currentResName = "end"
                            currentRotNum = 0
                        }
                    }
                    rotNum = currentRotNum
                }
                resName = currentResName
            }
            /*// Create variables for various parameters of interest
                        Character chainID = lineArr[3].toCharacter()
                        String chainName = lineArr[4]
                        int resID = lineArr[6].toInteger()
                        String resName = lineArr[5]
                        int rotNum = lineArr[9].toInteger()
                        Polymer polymer = activeAssembly.getChain(chainName)
                        Residue residue = polymer.getResidue(resID)

                        String currentResName = resName
                        int currentRotNum = rotNum
                        // While residue name remains the same
                        while(currentResName.equals(resName)){
                            ArrayList<String[]> atomsInRotamer = new ArrayList()
                            while(currentRotNum == rotNum){
                                String atomName = lineArr[10]
                                String x = lineArr[12]
                                String y = lineArr[13]
                                String z = lineArr[14]
                                String[] singleAtomCoords = [atomName, x, y, z]
                                atomsInRotamer.add(singleAtomCoords)

                                // read in next line and set new "current" variables
                                if(reader.readLine() != null){
                                    line = reader.readLine()
                                    lineArr = line.split(':')
                                    currentRotNum = lineArr[9].toInteger()
                                    currentResName = lineArr[5]
                                }
                            }
                            ArrayList<Atom> sideChainAtoms = residue.getSideChainAtoms()
                            logger.info("Got side chain atoms for residue "+currentResName) // Here for Val

                            // Loop over side chain atoms
                            for(Atom atom : sideChainAtoms){
                                String resAtomName = atom.getName()

                                // Match side chain atoms to atoms in rotamer coords array based on name
                                for(String[] coordsArray : atomsInRotamer){
                                    String coordsArrayAtomName = coordsArray[0]

                                    // If atom names match, set new coordinates based on coordinates in rotamer coords array
                                    if(coordsArrayAtomName.equals(resAtomName)){
                                        double x = Double.parseDouble(coordsArray[1])
                                        double y = Double.parseDouble(coordsArray[2])
                                        double z = Double.parseDouble(coordsArray[3])

                                        double[] xyz = [x,y,z] as double[]
                                        atom.setXYZ(xyz)
                                        logger.info("Added atom coordinates for "+atom.getName()+" in residue "+currentResName)
                                        // Store residue state with new coordinates for matched atom
                                        residue.storeState()
                                    }
                                }
                            } // End side chain atoms loop

                            // Create and set new rotamer for this residue
                            // Rotamer constructor wants an AminoAcid3 or NucleicAcid3 object
                            ResidueEnumerations.AminoAcid3 aminoAcid = residue.getAminoAcid3()
                            Rotamer rotamer = new Rotamer(aminoAcid)
                            residue.addRotamer(rotamer)
                            logger.info("Added rotamer for residue "+residue.getName()) // Here for Val
                        }*/

        } else{
            logger.warning("Could not read file")
        }

        // End rotamer adding code

        RotamerOptimization rotamerOptimization = new RotamerOptimization(
                activeAssembly, activeAssembly.getPotentialEnergy(), algorithmListener)

        testing = getTesting()
        if (testing) {
            rotamerOptimization.turnRotamerSingleEliminationOff()
            rotamerOptimization.turnRotamerPairEliminationOff()
        }

        if (monteCarloTesting) {
            rotamerOptimization.setMonteCarloTesting(true)
        }
        manyBody.initRotamerOptimization(rotamerOptimization, activeAssembly)

        ArrayList<Residue> residueList = rotamerOptimization.getResidues();

        boolean master = true;
        if (Comm.world().size() > 1) {
            int rank = Comm.world().rank();
            if (rank != 0) {
                master = false;
            }
        }

        algorithmFunctions.energy(activeAssembly)

        RotamerLibrary.measureRotamers(residueList, false);

        RotamerOptimization.Algorithm algo;
        switch (manyBody.getAlgorithmNumber()) {
            case 1:
                algo = RotamerOptimization.Algorithm.INDEPENDENT;
                break;
            case 2:
                algo = RotamerOptimization.Algorithm.ALL;
                break;
            case 3:
                algo = RotamerOptimization.Algorithm.BRUTE_FORCE;
                break;
            case 4:
                algo = RotamerOptimization.Algorithm.WINDOW;
                break;
            case 5:
                algo = RotamerOptimization.Algorithm.BOX;
                break;
            default:
                throw new IllegalArgumentException(String.format(" Algorithm choice was %d, not in range 1-5!", manyBody.getAlgorithmNumber()));
        }
        rotamerOptimization.optimize(algo);

        if (master) {
            logger.info(" Final Minimum Energy");

            File modelFile = saveDirFile(activeAssembly.getFile());
            algorithmFunctions.saveAsPDB(activeAssembly, modelFile);
            algorithmFunctions.energy(activeAssembly);
        }

        manyBody.saveEliminatedRotamers();

        if (priorGKwarn == null) {
            System.clearProperty("gk-suppressWarnings");
        }

        return this;
    }

    /**
     * Returns the potential energy of the active assembly. Used during testing assertions.
     * @return potentialEnergy Potential energy of the active assembly.
     */
    ForceFieldEnergy getPotential() {
        return potentialEnergy;
    }

    @Override
    public List<Potential> getPotentials() {
        return potentialEnergy == null ? Collections.emptyList() : Collections.singletonList(potentialEnergy);
    }

    /**
     * Set method for the testing boolean. When true, the testing boolean will shut off all elimination criteria forcing either a monte carlo or brute force search over all permutations.
     * @param testing A boolean flag that turns off elimination criteria for testing purposes.
     */
    void setTesting(boolean testing) {
        this.testing = testing;
    }

    /**
     * Get method for the testing boolean. When true, the testing boolean will shut off all elimination criteria forcing either a monte carlo or brute force search over all permutations.
     * @return testing A boolean flag that turns off elimination criteria for testing purposes.
     */
    boolean getTesting() {
        return testing;
    }

    /**
     * Set to true when testing the monte carlo rotamer optimization algorithm. True will trigger the "set seed"
     * functionality of the pseudo-random number generator in the RotamerOptimization.java class to create a deterministic monte carlo algorithm.
     * @param bool True ONLY when a deterministic monte carlo approach is desired. False in all other cases.
     */
    void setMonteCarloTesting(boolean bool) {
        this.monteCarloTesting = bool;
    }
}

/**
 * Title: Force Field X.
 *
 * Description: Force Field X - Software for Molecular Biophysics.
 *
 * Copyright: Copyright (c) Michael J. Schnieders 2001-2018.
 *
 * This file is part of Force Field X.
 *
 * Force Field X is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as published by
 * the Free Software Foundation.
 *
 * Force Field X is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Force Field X; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Linking this library statically or dynamically with other modules is making a
 * combined work based on this library. Thus, the terms and conditions of the
 * GNU General Public License cover the whole combination.
 *
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

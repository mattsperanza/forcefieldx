//******************************************************************************
//
// Title:       Force Field X.
// Description: Force Field X - Software for Molecular Biophysics.
// Copyright:   Copyright (c) Michael J. Schnieders 2001-2023.
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
//******************************************************************************
package ffx.algorithms.groovy.test

import ffx.algorithms.cli.AlgorithmsScript
import ffx.algorithms.optimize.ConformationScan
import ffx.potential.ForceFieldEnergy
import ffx.potential.MolecularAssembly
import ffx.potential.bonded.Atom
import ffx.potential.bonded.Molecule
import ffx.potential.utils.PotentialsUtils
import org.apache.commons.io.FilenameUtils
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import picocli.CommandLine.Parameters

/**
 * GenerateCrystalSeeds is a Groovy script that generates a set of molecular orientations in vacuum and
 * calculates the energy of each conformation.
 * <br>
 * Usage:
 * <br>
 * ffxc test.GenerateCrystalSeeds [options] &lt;filename&gt;
 */
@Command(description = " Calculates interaction energies of different molecular orientations and saves low energy orientations.",
        name = "test.GenerateCrystalSeeds")
class CoformerBindingSearch extends AlgorithmsScript {

    /**
     * --eps
     */
    @Option(names = ['--eps'], paramLabel = '.1',
            description = 'Gradient cutoff for minimization.')
    double eps = 0.1

    /**
     * --maxIter
     */
    @Option(names = ['--maxIter'], paramLabel = '1000',
            description = 'Max iterations for minimization.')
    int maxIter = 1000

    /**
     * --hBondDist
     */
    @Option(names = ['--hBondDist', '--hbd'], paramLabel = '2.0',
            description = 'Initial h-bond distance in angstroms.')
    double hBondDist = 2.0

    /**
     * --flatBottomRadius
     */
    @Option(names = ['--flatBottomRadius', "--fbr"], paramLabel = '0.5',
            description = 'Radius of flat bottom bond restraint potential in angstroms.')
    double flatBottomRadius = 0.5

    /**
     * --gkSoluteDielectric
     */
    @Option(names = ['--gkSolventDielectric'], paramLabel = '78.4',
            description = 'Sets the gk solvent dielectric constant.')
    double gkSolventDielec = 78.4

    /**
     * --skipHomodimerNumber
     */
    @Option(names = ['--skipHomodimerNumber'], paramLabel = '-1',
            description = 'Skip conformation search on input dimer one or two.')
    int skipHomodimerNumber = -1

    /**
     * --intermediateTorsionScan
     */
    @Option(names = ['--intermediateTorsionScan', "--its"], paramLabel = 'false', defaultValue = 'false',
            description = 'During sampling, statically scan torsions after direct minimization to find the lowest energy conformation.')
    private boolean intermediateTorsionScan = false

    /**
     * --noMinimize
     */
    @Option(names = ['--noMinimize'], paramLabel = 'false', defaultValue = 'false',
            description = 'Don\'t minimize or torsion scan after conformations are generated. Useful for testing.')
    private boolean noMinimize = false

    /**
     * --excludeH
     */
    @Option(names = ['--excludeH', "--eh"], paramLabel = 'false', defaultValue = 'false',
            description = 'Only include H bonded to electronegative atoms in conformations.')
    private boolean excludeH = false

    /**
     * --coformerOnly
     */
    @Option(names = ['--coformerOnly'], paramLabel = 'false', defaultValue = 'false',
            description = 'Only conformation search the coformer.')
    private boolean coformerOnly = false

    /**
     * --gk
     */
    @Option(names = ['--gk'], paramLabel = 'false', defaultValue = 'false',
            description = 'Use generalized kirkwood solvent.')
    private boolean gk = false

    /**
     * Filename.
     */
    @Parameters(arity = "1..*", paramLabel = "files",
            description = "XYZ input file.")
    private List<String> filenames

    /**
     * Constructor.
     */
    CoformerBindingSearch() {
        this(new Binding())
    }

    /**
     * Constructor.
     * @param binding The Groovy Binding to use.
     */
    CoformerBindingSearch(Binding binding) {
        super(binding)
    }

    /**
     * {@inheritDoc}
     */
    @Override
    CoformerBindingSearch run() {
        // Init the context and bind variables.
        if (!init()) {
            return this
        }

        // Cat the key files together and set the -Dkey property to be the new file we created
        // Write default gk options to the key file
        // Only does a simple search for the patch file so it needs to be named accordingly with the .xyz
        setKeyAndPatchFilesProperly(gk, gkSolventDielec, filenames)

        // Check the size of the filenames list
        if (!(filenames.size() == 1 || filenames.size() == 2)) {
            logger.severe("Must provide one or two filenames.")
            return this
        }

        boolean minimize = !noMinimize

        // Load the MolecularAssembly of the input file.
        PotentialsUtils potentialsUtils = new PotentialsUtils()
        boolean skipMoleculeOne = skipHomodimerNumber == 1
        boolean skipMoleculeTwo = skipHomodimerNumber == 2

        // Perform scan on monomer one w/ itself
        ConformationScan monomerOneScan = null
        ConformationScan monomerTwoScan = null
        if(!coformerOnly) {
            if (!skipMoleculeOne) {
                MolecularAssembly[] molecularAssemblies = potentialsUtils.openAll(new String[]{filenames.get(0), filenames.get(0)})
                MolecularAssembly combined = combineTwoMolecularAssembliesWOneMolEach(molecularAssemblies[0], molecularAssemblies[1])
                Molecule[] molecules = combined.getMolecules()
                monomerOneScan = new ConformationScan(
                        combined,
                        molecules[0],
                        molecules[1],
                        eps,
                        maxIter,
                        hBondDist,
                        flatBottomRadius,
                        intermediateTorsionScan,
                        excludeH,
                        minimize
                )
                monomerOneScan.scan()
                logger.info("\n Molecule one (" + FilenameUtils.removeExtension(filenames.get(0)) + ") dimer scan energy information:")
                monomerOneScan.logAllEnergyInformation()
                String molOneDimerScanFilename = FilenameUtils.removeExtension(filenames.get(0)) + ".arc"
                File molOneDimerScanFile = new File(molOneDimerScanFilename)
                monomerOneScan.writeStructuresToXYZ(molOneDimerScanFile)
            } else{
                logger.info(" Skipping monomer one scan.")
            }

            if (!skipMoleculeTwo && filenames.size() == 2) {
                MolecularAssembly[] molecularAssemblies = potentialsUtils.openAll(new String[]{filenames.get(1), filenames.get(1)})
                MolecularAssembly combined = combineTwoMolecularAssembliesWOneMolEach(molecularAssemblies[0], molecularAssemblies[1])
                Molecule[] molecules = combined.getMolecules()
                monomerTwoScan = new ConformationScan(
                        combined,
                        molecules[0],
                        molecules[1],
                        eps,
                        maxIter,
                        hBondDist,
                        flatBottomRadius,
                        intermediateTorsionScan,
                        excludeH,
                        minimize
                )
                monomerTwoScan.scan()
                logger.info("\n Molecule two (" + FilenameUtils.removeExtension(filenames.get(1)) + ") dimer scan energy information:")
                monomerTwoScan.logAllEnergyInformation()
                String molTwoDimerScanFilename = FilenameUtils.removeExtension(filenames.get(1)) + ".arc"
                File molTwoDimerScanFile = new File(molTwoDimerScanFilename)
                monomerTwoScan.writeStructuresToXYZ(molTwoDimerScanFile)
            } else if (!skipMoleculeTwo && filenames.size() == 1) {
                logger.info(" Only one file provided, skipping second homodimer scan.")
            }
        } else {
            logger.info(" Skipping monomer one and two scan.")
        }

        if(filenames.size() == 2){
            MolecularAssembly[] molecularAssemblies = potentialsUtils.openAll(new String[]{filenames.get(0), filenames.get(1)})
            MolecularAssembly combined = combineTwoMolecularAssembliesWOneMolEach(molecularAssemblies[0], molecularAssemblies[1])
            Molecule[] molecules = combined.getMolecules()
            // Smaller molecule is molecule one
            Molecule mol1 = molecules[0].atomList.size() < molecules[1].atomList.size() ? molecules[0] : molecules[1]
            Molecule mol2 = molecules[0].atomList.size() < molecules[1].atomList.size() ? molecules[1] : molecules[0]
            ConformationScan dimerScan = new ConformationScan(
                    combined,
                    mol1,
                    mol2,
                    eps,
                    maxIter,
                    hBondDist,
                    flatBottomRadius,
                    intermediateTorsionScan,
                    excludeH,
                    minimize
            )
            dimerScan.scan()
            logger.info("\n Molecule one (" + FilenameUtils.removeExtension(filenames.get(0)) +
                    ") and two (" + FilenameUtils.removeExtension(filenames.get(1)) + ") dimer scan energy information:")
            dimerScan.logAllEnergyInformation()
            File coformerScanFile = new File("coformerScan.arc")
            dimerScan.writeStructuresToXYZ(coformerScanFile)

            if(monomerOneScan != null && monomerTwoScan != null){
                logger.info("\n Molecule one (" + FilenameUtils.removeExtension(filenames.get(0)) + ") dimer scan energy information:")
                monomerOneScan.logAllEnergyInformation()
                logger.info("\n Molecule two (" + FilenameUtils.removeExtension(filenames.get(1)) + ") dimer scan energy information:")
                monomerTwoScan.logAllEnergyInformation()
                logger.info("\n Molecule one (" + FilenameUtils.removeExtension(filenames.get(0)) +
                        ") and two (" + FilenameUtils.removeExtension(filenames.get(1)) + ") dimer scan energy information:")
                dimerScan.logAllEnergyInformation()
                ConformationScan.logBindingEnergyCalculation(monomerOneScan, monomerTwoScan, dimerScan)
            }
        }
        else{
            logger.info(" Only one file provided, skipping coformer scan.")
        }

        return this
    }

    static MolecularAssembly combineTwoMolecularAssembliesWOneMolEach(MolecularAssembly mola1, MolecularAssembly mola2){
        MolecularAssembly mainMonomerAssembly = mola1
        MolecularAssembly feederAssembly = mola2
        Molecule[] assemblyTwoMolecules = feederAssembly.getMoleculeArray()
        for(Atom a: assemblyTwoMolecules[0].getAtomList()) {
            a.setMoleculeNumber(1)
            a.move(new double[]{10,-10,10})
        }
        assemblyTwoMolecules[0].setName("Molecule-2")
        mainMonomerAssembly.addMSNode(assemblyTwoMolecules[0])
        mainMonomerAssembly.update()
        mainMonomerAssembly.setPotential(null) // energyFactory doesn't do anything if it isn't null
        ForceFieldEnergy forceFieldEnergy = ForceFieldEnergy.energyFactory(mainMonomerAssembly)
        return mainMonomerAssembly
    }

    static void setKeyAndPatchFilesProperly(boolean gk, double gkSolventDielec, List<String> filenames) {
        String key = "coformerScan.key"
        String patch = "coformerScan.patch"
        // Create the key file
        File keyFile = new File(key)
        File patchFile = new File(patch)
        logger.info(" Creating key file: " + key)
        keyFile.createNewFile()
        // concatenate the two files together with bufferedReader and bufferedWriter
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(keyFile))
            bw.write("patch " + FilenameUtils.removeExtension(key) + ".patch\n\n")
            if(gk){
                bw.write("gkterm true\n")
                bw.write("solvent-dielectric " + gkSolventDielec + "\n")
                bw.write("gk-radius solute\n")
                bw.write("cavmodel gauss-disp\n")
            }
            bw.close()
        } catch (IOException e) {
            e.printStackTrace()
        }
        logger.info(" Creating patch file: " + patch)
        patchFile.createNewFile()
        String patchOne = FilenameUtils.removeExtension(filenames.get(0)) + ".patch"
        String patchTwo = FilenameUtils.removeExtension(filenames.get(1)) + ".patch"
        String[] files = new String[]{patchOne, patchTwo}
        BufferedWriter bw = new BufferedWriter(new FileWriter(patchFile))
        for (String file : files) {
            BufferedReader br = new BufferedReader(new FileReader(file))
            String line = br.readLine()
            while (line != null) {
                bw.write(line + "\n")
                line = br.readLine()
            }
            br.close()
        }
        bw.close()
        System.setProperty("key", key)
    }
}


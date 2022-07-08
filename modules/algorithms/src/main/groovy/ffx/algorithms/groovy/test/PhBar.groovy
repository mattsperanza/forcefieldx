//******************************************************************************
//
// Title:       Force Field X.
// Description: Force Field X - Software for Molecular Biophysics.
// Copyright:   Copyright (c) Michael J. Schnieders 2001-2021.
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


import edu.rit.mp.DoubleBuf
import edu.rit.pj.Comm
import ffx.algorithms.cli.AlgorithmsScript
import ffx.algorithms.cli.DynamicsOptions
import ffx.algorithms.dynamics.MolecularDynamics
import ffx.algorithms.dynamics.MolecularDynamicsOpenMM
import ffx.numerics.Potential
import ffx.potential.bonded.Residue
import ffx.potential.cli.WriteoutOptions
import ffx.potential.extended.ExtendedSystem
import picocli.CommandLine.Command
import picocli.CommandLine.Mixin
import picocli.CommandLine.Option
import picocli.CommandLine.Parameters

import java.util.logging.Level

import static java.lang.String.format
/**
 * The Thermodynamics script uses the BAR algorithm to estimate a free energy.
 * <br>
 * Usage:
 * <br>
 * ffxc Thermodynamics [options] &lt;filename&gt [file2...];
 */
@Command(description = " Use the BAR algorithm to estimate a free energy.", name = "ffxc PhBar")
class PhBar extends AlgorithmsScript {

    @Mixin
    DynamicsOptions dynamicsOptions

    @Mixin
    WriteoutOptions writeOutOptions

    /**
     * --pH or --constantPH Constant pH value for molecular dynamics.
     */
    @Option(names = ['--pH', '--constantPH'], paramLabel = '7.4',
            description = 'Constant pH value for molecular dynamics windows')
    double pH = 7.4

    /**
     * --titrationFix Constant titration state for windows [0-10]
     */
    @Option(names = ['--titrationFix'], paramLabel = '0',
            description = 'Constant titration value for molecular dynamics windows')
    double fixedTitrationState = -1

    /**
     * --tautomerFix Constant tautomer state for windows [0-10]
     */
    @Option(names = ['--tautomerFix'], paramLabel = '0',
            description = 'Constant tautomer value for molecular dynamics windows')
    double fixedTautomerState = -1

    @Option(names = ['--iterations'], paramLabel = '999',
            description = 'Number of times to evaluate neighbor energies')
    int cycles  = 999

    @Option(names = ['--coordinateSteps'], paramLabel = '10000',
            description = 'Number of steps done on GPU before each evaluation')
    int coordSteps  = 10000

    /**
     * One or more filenames.
     */
    @Parameters(arity = "1..*", paramLabel = "files",
            description = "XYZ or PDB input files.")
    private String filename


    private Potential forceFieldEnergy
    MolecularDynamics molecularDynamics = null
    boolean lockTitration = false
    boolean lockTautomer = false
    double energy
    ArrayList<Double> previous = new ArrayList<>()
    ArrayList<Double> current = new ArrayList<>()
    ArrayList<Double> next = new ArrayList<>()


    /**
     * Thermodynamics Constructor.
     */
    PhBar() {
        this(new Binding())
    }

    /**
     * Thermodynamics Constructor.
     * @param binding The Groovy Binding to use.
     */
    PhBar(Binding binding) {
        super(binding)
    }

    PhBar run(){

        if (!init()) {
            return this
        }

        Comm world = Comm.world()
        int nRanks = world.size()
        if(nRanks < 11){
            logger.warning(" Running BAR with less then the usual amount of windows")
        }
        int myRank = (nRanks > 1) ? world.rank() : 0

        // Init titration states and decide what is being locked
        if(fixedTitrationState == -1 && fixedTautomerState == -1) {
            logger.severe(" Must select a tautomer or titration to fix windows at. The program will not continue")
            return this
        } else if(fixedTautomerState != -1){
            fixedTitrationState = (double) myRank / nRanks
            lockTautomer = true
            logger.info(" Running BAR across titration states with tautomer state locked at " + fixedTautomerState)
            logger.info(" Titration state for this rank: " + fixedTitrationState)
        } else if (fixedTitrationState != -1){
            fixedTautomerState = (double) myRank / nRanks
            lockTitration = true
            logger.info(" Running BAR across tautomer states with titration state locked at " + fixedTitrationState)
            logger.info(" Tautomer state for this rank: " + fixedTautomerState)
        }

        // Illegal state checks
        if(fixedTautomerState > 1 || fixedTitrationState > 1){
            logger.severe(" ERROR: Cannot assign lambda state to > 1")
            return this
        }

        if(fixedTautomerState < 0 || fixedTitrationState < 0){
            logger.severe(" ERROR: Cannot assign lambda state to < 1")
            return this
        }

        dynamicsOptions.init()

        activeAssembly = getActiveAssembly(filename)
        if (activeAssembly == null) {
            logger.info(helpString())
            return this
        }
        forceFieldEnergy = activeAssembly.getPotentialEnergy()

        // Set the filename.
        String filename = activeAssembly.getFile().getAbsolutePath()

        //TODO: Restart Stuff?
        /*
        // Restart File
        File esv = new File(FilenameUtils.removeExtension(filename) + ".esv")
        if (!esv.exists()) {
            esv = null
        }*/
        /*
        // Restart File
        File dyn = new File(FilenameUtils.removeExtension(filename) + ".dyn")
        if (!dyn.exists()) {
            dyn = null
        }*/

        // Initialize and attach extended system first.
        ExtendedSystem esvSystem = new ExtendedSystem(activeAssembly, null)

        //Setting the systems locked states
        for(Residue res: esvSystem.getExtendedResidueList()){
            esvSystem.setTitrationLambda(res, fixedTitrationState)
            if(esvSystem.isTautomer(res)){
                esvSystem.setTautomerLambda(res, fixedTautomerState)
            }
        }

        esvSystem.setConstantPh(pH)
        esvSystem.setFixTitrationState(true)
        esvSystem.setFixTautomerState(true)
        forceFieldEnergy.attachExtendedSystem(esvSystem)

        int numESVs = esvSystem.extendedResidueList.size()
        logger.info(format(" Attached extended system with %d residues.", numESVs))

        double[] x = new double[forceFieldEnergy.getNumberOfVariables()]
        forceFieldEnergy.getCoordinates(x)
        forceFieldEnergy.energy(x, true)

        logger.info("\n Running molecular dynamics on " + filename)

        molecularDynamics = dynamicsOptions.getDynamics(writeOutOptions, forceFieldEnergy, activeAssembly, algorithmListener)

        File structureFile = new File(filename)
        File rankDirectory = new File(structureFile.getParent() + File.separator + Integer.toString(myRank))
        if (!rankDirectory.exists()) {
            rankDirectory.mkdir()
        }
        final String newMolAssemblyFile = rankDirectory.getPath() + File.separator + structureFile.getName()
        logger.info(" Set activeAssembly filename: " + newMolAssemblyFile)
        activeAssembly.setFile(new File(newMolAssemblyFile))

        if (molecularDynamics instanceof MolecularDynamicsOpenMM) {
            MolecularDynamicsOpenMM molecularDynamicsOpenMM = molecularDynamics

            molecularDynamics = dynamicsOptions.getDynamics(writeOutOptions, forceFieldEnergy, activeAssembly, algorithmListener,
                    MolecularDynamics.DynamicsEngine.FFX)
            molecularDynamics.attachExtendedSystem(esvSystem, dynamicsOptions.report)

            for (int i = 0; i < cycles; i++) {
                logger.info(" ________________________CYCLE " + i + "____________________________")
                molecularDynamics.setCoordinates(x)
                molecularDynamics.dynamic(1, dynamicsOptions.dt, 1, dynamicsOptions.write,
                        dynamicsOptions.temperature, true, null)

                molecularDynamicsOpenMM.setCoordinates(x)
                forceFieldEnergy.energy(x)

                molecularDynamicsOpenMM.dynamic(coordSteps, dynamicsOptions.dt, dynamicsOptions.report, dynamicsOptions.write,
                        dynamicsOptions.temperature, true, null)
                x = molecularDynamicsOpenMM.getCoordinates()

                double titrationNeighbor = lockTitration ? 0 : (double) 1 / nRanks
                double tautomerNeighbor = lockTautomer ? 0 : (double) 1 / nRanks

                logger.info("Rank: " + myRank + "  Titration Perturbation: " + titrationNeighbor + "  Tautomer Perturbation: " + tautomerNeighbor)
                logger.info("List of Residues: " + esvSystem.getExtendedResidueList())
                logger.info("List of Titrating Residues: " + esvSystem.getTitratingResidueList())
                logger.info("List of Tautomerizing Residues" + esvSystem.getTautomerizingResidueList())

                if(myRank != nRanks-1) {
                    for (Residue res : esvSystem.getExtendedResidueList()) {
                        esvSystem.setTitrationLambda(res, fixedTitrationState + titrationNeighbor, false)
                        //esvSystem.perturbLambdas(lockTautomer, fixedTitrationState + titrationNeighbor)
                        if (esvSystem.isTautomer(res)) {
                            esvSystem.setTautomerLambda(res, fixedTautomerState + tautomerNeighbor, false)
                            //esvSystem.perturbLambdas(lockTautomer, fixedTautomerState + tautomerNeighbor)
                        }
                    }
                    forceFieldEnergy.getCoordinates(x)
                    energy = forceFieldEnergy.energy(x, false)
                    next.add(energy)
                }

                if(myRank != 0) {
                    for (Residue res : esvSystem.getExtendedResidueList()) {
                        esvSystem.setTitrationLambda(res, fixedTitrationState - titrationNeighbor, false)
                        //esvSystem.perturbLambdas(lockTautomer, fixedTautomerState - titrationNeighbor)
                        if (esvSystem.isTautomer(res)) {
                            esvSystem.setTautomerLambda(res, fixedTautomerState - tautomerNeighbor, false)
                            //esvSystem.perturbLambdas(lockTautomer, fixedTautomerState - tautomerNeighbor)
                        }
                    }
                    forceFieldEnergy.getCoordinates(x)
                    energy = forceFieldEnergy.energy(x, false)
                    previous.add(energy)
                }

                for(Residue res: esvSystem.getExtendedResidueList()){
                    esvSystem.setTitrationLambda(res, fixedTitrationState, false)
                    //esvSystem.perturbLambdas(lockTautomer, fixedTitrationState)
                    if(esvSystem.isTautomer(res)){
                        esvSystem.setTautomerLambda(res, fixedTautomerState, false)
                        //esvSystem.perturbLambdas(lockTautomer, fixedTautomerState)
                    }
                }
                forceFieldEnergy.getCoordinates(x)
                energy = forceFieldEnergy.energy(x, false)
                current.add(energy)
            }
        }else {
            logger.severe(" MD is not an instance of MDOMM")
        }

        if(current.size() != cycles){
            logger.severe(" Size of the self energies array is not equal to number of cycles")
            return this
        }

        logger.info(" ________________________End of Cycles for rank: " + myRank + "___________________________")

        double[][] parameters = new double[nRanks][current.size() * 3]
        DoubleBuf[] parametersBuf = new DoubleBuf[nRanks];
        int counter = 0
        if(myRank != 0) {
            for (int i = 0; i < previous.size(); i++) {
                parameters[myRank][i] = previous.get(i)
                counter = i
            }
        }
        for (int i = counter; i < counter + current.size(); i++){
            parameters[myRank][i - counter] = current.get(i - counter)
            counter = i
        }
        if(myRank != nRanks - 1) {
            for (int i = counter; i < counter + next.size(); i++) {
                parameters[myRank][i - counter] = previous.get(i - counter)
            }
        }

        for (int i = 0; i < nRanks; i++) {
            parametersBuf[i] = DoubleBuf.buffer(parameters[i])
        }

        DoubleBuf myParametersBuf = parametersBuf[myRank]


        try {
            world.allGather(myParametersBuf, parametersBuf)
        } catch (IOException ex) {
            String message = " CreateBAR allGather failed."
            logger.log(Level.SEVERE, message, ex)
        }

        logger.log(" ______________________________All Gather Complete________________________")
        if(myRank != nRanks - 1) {

            File outputDir = new File(rankDirectory.getParent() + File.separator + "barFiles")
            if (!outputDir.exists()) {
                outputDir.mkdir()
            }
            File output = new File(outputDir.getPath() + File.separator + "energy_" + myRank + ".bar")

            try (FileWriter fw = new FileWriter(output)
                 BufferedWriter bw = new BufferedWriter(fw)) {
                bw.write(format("    %d  %f  this.xyz", current.size(), dynamicsOptions.temperature))

                for (int i = 1; i <= current.size(); i++){
                    bw.write(format("%5d%17.9f%17.9f", i, parameters[myRank][current.size() + i - 1], parameters[myRank][current.size() * 2 + i - 2]))
                }

                for (int i = 1; i <= current.size(); i++){
                    bw.write(format("%5d%17.9f%17.9f", i, parameters[myRank + 1][i - 1], parameters[myRank + 1][current.size() + i - 1]))
                }
            } catch (IOException e) {
                e.printStackTrace()
            }
        }

        return this
    }
}

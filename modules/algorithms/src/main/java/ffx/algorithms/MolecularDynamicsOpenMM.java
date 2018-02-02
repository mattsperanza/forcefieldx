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
package ffx.algorithms;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import static java.lang.String.format;

import com.sun.jna.ptr.PointerByReference;

import org.apache.commons.configuration.CompositeConfiguration;
import org.apache.commons.io.FilenameUtils;

import static simtk.openmm.AmoebaOpenMMLibrary.OpenMM_KcalPerKJ;
import static simtk.openmm.OpenMMLibrary.OpenMM_Context_getState;
import static simtk.openmm.OpenMMLibrary.OpenMM_Context_setVelocitiesToTemperature;
import static simtk.openmm.OpenMMLibrary.OpenMM_Integrator_step;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_DataType.OpenMM_State_Energy;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_DataType.OpenMM_State_Forces;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_DataType.OpenMM_State_Positions;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_DataType.OpenMM_State_Velocities;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_destroy;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_getForces;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_getKineticEnergy;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_getPositions;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_getPotentialEnergy;
import static simtk.openmm.OpenMMLibrary.OpenMM_State_getVelocities;

import ffx.algorithms.integrators.IntegratorEnum;
import ffx.algorithms.thermostats.ThermostatEnum;
import ffx.crystal.Crystal;
import ffx.potential.ForceFieldEnergyOpenMM;
import ffx.potential.MolecularAssembly;
import ffx.potential.bonded.Atom;
import ffx.potential.extended.ExtendedSystem;
import ffx.potential.parsers.DYNFilter;
import ffx.potential.parsers.PDBFilter;
import ffx.potential.parsers.XYZFilter;
import ffx.potential.utils.PotentialsFunctions;
import ffx.potential.utils.PotentialsUtils;
import static ffx.algorithms.thermostats.Thermostat.convert;
import static ffx.algorithms.thermostats.Thermostat.kB;

/**
 * Runs Molecular Dynamics using OpenMM implementation
 *
 * @author Hernan V. Bernabe
 */
public class MolecularDynamicsOpenMM extends MolecularDynamics {

    private static final Logger logger = Logger.getLogger(MolecularDynamicsOpenMM.class.getName());

    /**
     * OpenMM ForceFieldEnergy.
     */
    private ForceFieldEnergyOpenMM forceFieldEnergyOpenMM;
    /**
     * OpenMM Context.
     */
    private PointerByReference context;
    /**
     * OpenMM Integrator.
     */
    private PointerByReference integrator;
    /**
     * OpenMM Positions.
     */
    private PointerByReference positions;
    /**
     * OpenMM Velocities.
     */
    private PointerByReference velocities;
    /**
     * OpenMM Forces.
     */
    private PointerByReference forces;
    /**
     * Number of OpenMM Particles (i.e. the number of FFX atoms).
     */
    private int numParticles;
    /**
     * Number of OpenMM Degrees of Freedom.
     */
    private int dof;
    /**
     * Integrator Type.
     */
    private final IntegratorEnum integratorType;
    /**
     * Thermostat Type.
     */
    private final ThermostatEnum thermostatType;
    /**
     * Integrator String.
     */
    private String integratorString;
    /**
     * Thermostat String.
     */
    private String thermostatString;
    /**
     * Dynamics State. ToDo: the DynamicsState logic can probably be made consistent with an OpenMM State.
     */
    private DynamicsState lastState = new DynamicsState();
    /**
     * Number of OpenMM MD steps per iteration.
     */
    private int intervalSteps;
    /**
     * Random number generation.
     */
    private Random random;
    /**
     * Andersen thermostat collision frequency.
     */
    private double collisionFreq = 0.01;
    /**
     * Flag to indicate OpenMM MD interactions are running.
     */
    private boolean running;
    /**
     * Run time.
     */
    private long time;
    /**
     * Number of Infinite Energy values.
     */
    private int numInfiniteEnergies = 0;
    
    private long mdTime = 0;
    
    private double startingTotalEnergy;
    
    private double endTotalEnergy;

    /**
     * Constructs an MolecularDynamicsOpenMM object, to perform molecular
     * dynamics using native OpenMM routines, avoiding the cost of communicating
     * coordinates, gradients, and energies back and forth across the PCI bus.
     *
     * @param assembly MolecularAssembly to operate on
     * @param forceFieldEnergyOpenMM ForceFieldEnergyOpenMM Potential. Cannot be
     * any other type of Potential.
     * @param properties Associated properties
     * @param listener
     * @param thermostat May have to be slightly modified for native OpenMM
     * routines
     * @param integratorMD May have to be slightly modified for native OpenMM
     * routines
     */
    public MolecularDynamicsOpenMM(MolecularAssembly assembly, ForceFieldEnergyOpenMM forceFieldEnergyOpenMM,
                                   CompositeConfiguration properties, AlgorithmListener listener,
                                   ThermostatEnum thermostat, IntegratorEnum integratorMD) {
        super(assembly, forceFieldEnergyOpenMM, properties, listener, thermostat, integratorMD);

        /**
         * Initialization specific to MolecularDynamicsOpenMM
         */
        this.forceFieldEnergyOpenMM = forceFieldEnergyOpenMM;
        this.numParticles = forceFieldEnergyOpenMM.getNumParticles();
        forceFieldEnergyOpenMM.addCOMMRemover(false);
        random = new Random();
        this.thermostatType = thermostat;
        this.integratorType = integratorMD;

        running = false;

        integratorToString(integratorMD);
        updateIntegrator();

    }

    /**
     * UNSUPPORTED: MolecularDynamicsOpenMM is not presently capable of handling
     * extended system variables. Will throw an UnsupportedOperationException.
     *
     * @param system
     * @param printFrequency
     */
    @Override
    public void attachExtendedSystem(ExtendedSystem system, int printFrequency) {
        throw new UnsupportedOperationException(" MolecularDynamicsOpenMM does not support extended system variables!");
    }

    /**
     * UNSUPPORTED: MolecularDynamicsOpenMM is not presently capable of handling
     * extended system variables. Will throw an UnsupportedOperationException.
     */
    @Override
    public void detachExtendedSystem() {
        throw new UnsupportedOperationException(" MolecularDynamicsOpenMM does not support extended system variables!");
    }

    /**
     * takeSteps moves the simulation forward in time a user defined number of
     * steps and integrates the equations of motion for each step. This method
     * ensures that the algorithm reports back only when the time interval
     * (steps) specified by the user is completed.
     *
     * @param intervalSteps
     */
    private void takeSteps(int intervalSteps) {
        OpenMM_Integrator_step(integrator, intervalSteps);
    }

    /**
     * openMM_Update obtains the state of the simulation from OpenMM, getting
     * positions and velocities back from the OpenMM data structure.
     *
     * @param i
     */
    private void openMM_Update(int i, boolean running) {
        context = forceFieldEnergyOpenMM.getContext();
        int infoMask = OpenMM_State_Positions + OpenMM_State_Velocities + OpenMM_State_Forces + OpenMM_State_Energy;

        PointerByReference state = OpenMM_Context_getState(context, infoMask, forceFieldEnergyOpenMM.enforcePBC);
        currentPotentialEnergy = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ;
        currentKineticEnergy = OpenMM_State_getKineticEnergy(state) * OpenMM_KcalPerKJ;
        currentTotalEnergy = currentPotentialEnergy + currentKineticEnergy;
        currentTemperature = 2.0 * currentKineticEnergy * convert / (kB * dof);

        positions = OpenMM_State_getPositions(state);
        forceFieldEnergyOpenMM.getOpenMMPositions(positions, numParticles, x);

        velocities = OpenMM_State_getVelocities(state);
        forceFieldEnergyOpenMM.getOpenMMVelocities(velocities, numParticles, v);

        forces = OpenMM_State_getForces(state);
        forceFieldEnergyOpenMM.getOpenMMAccelerations(forces, numParticles, mass, a);
        forceFieldEnergyOpenMM.getOpenMMAccelerations(forces, numParticles, mass, aPrevious);

        if (!Double.isFinite(currentPotentialEnergy)) {
            logger.warning(" Non-finite energy returned by OpenMM: simulation probably unstable.");
            DynamicsState thisState = new DynamicsState();
            thisState.storeState();
            lastState.revertState();

            File origFile = molecularAssembly.getFile();
            String timeString = LocalDateTime.now().format(DateTimeFormatter.
                    ofPattern("yyyy_MM_dd-HH_mm_ss"));

            String filename = String.format("%s-LAST-%s.pdb",
                    FilenameUtils.removeExtension(molecularAssembly.getFile().getName()),
                    timeString);

            PotentialsFunctions ef = new PotentialsUtils();
            filename = ef.versionFile(filename);
            logger.info(String.format(" Writing before-error snapshot to file %s", filename));
            ef.saveAsPDB(molecularAssembly, new File(filename));
            molecularAssembly.setFile(origFile);

            thisState.revertState();

            filename = String.format("%s-ERROR-%s.pdb",
                    FilenameUtils.removeExtension(molecularAssembly.getFile().getName()),
                    timeString);

            filename = ef.versionFile(filename);
            logger.info(String.format(" Writing after-error snapshot to file %s", filename));
            ef.saveAsPDB(molecularAssembly, new File(filename));
            molecularAssembly.setFile(origFile);
            // Logging and printing here.
            if (numInfiniteEnergies++ > 200) {
                logger.severe(String.format(" %d infinite energies experienced; shutting down FFX", numInfiniteEnergies));
            }
        }

        lastState.storeState();

        if (running) {
            if (i == 0) {
                logger.info(format("\n  %8s %12s %12s %12s %8s %8s", "Time", "Kinetic", "Potential", "Total", "Temp", "CPU"));
                logger.info(format("  %8s %12s %12s %12s %8s %8s", "psec", "kcal/mol", "kcal/mol", "kcal/mol", "K", "sec"));
                logger.info(format("  %8s %12.4f %12.4f %12.4f %8.2f",
                        "", currentKineticEnergy, currentPotentialEnergy, currentTotalEnergy, currentTemperature));
                
                startingTotalEnergy = currentTotalEnergy;
                
            } else if (i % printFrequency == 0) {
                double simTime = i * dt * 1.0e-3;
                time += System.nanoTime();
                mdTime = time;
                logger.info(format(" %7.3e %12.4f %12.4f %12.4f %8.2f %8.2f",
                        simTime, currentKineticEnergy, currentPotentialEnergy,
                        currentTotalEnergy, currentTemperature, time * NS2SEC));
                
                endTotalEnergy = currentTotalEnergy;
                
                time = -System.nanoTime();
            }

            if (saveSnapshotFrequency > 0 && i % (saveSnapshotFrequency * 1000) == 0 && i != 0) {
                for (AssemblyInfo ai : assemblies) {
                    if (ai.archiveFile != null && !saveSnapshotAsPDB) {
                        if (ai.xyzFilter.writeFile(ai.archiveFile, true)) {
                            logger.info(String.format(" Appended snap shot to %s", ai.archiveFile.getName()));
                        } else {
                            logger.warning(String.format(" Appending snap shot to %s failed", ai.archiveFile.getName()));
                        }
                    } else if (saveSnapshotAsPDB) {
                        if (ai.pdbFilter.writeFile(ai.pdbFile, false)) {
                            logger.info(String.format(" Wrote PDB file to %s", ai.pdbFile.getName()));
                        } else {
                            logger.warning(String.format(" Writing PDB file to %s failed.", ai.pdbFile.getName()));
                        }
                    }
                }
            }

            /**
             * Write out restart files every saveRestartFileFrequency steps.
             */
            if (restartFrequency > 0 && i % (restartFrequency * 1000) == 0 && i != 0) {
                if (dynFilter.writeDYN(restartFile, molecularAssembly.getCrystal(), x, v, a, aPrevious)) {
                    logger.info(String.format(" Wrote dynamics restart file to " + restartFile.getName()));
                } else {
                    logger.info(String.format(" Writing dynamics restart file to " + restartFile.getName() + " failed"));
                }
            }
        }

        OpenMM_State_destroy(state);
    }

    @Override
    public void init(int numSteps, double timeStep, double printInterval, double saveInterval,
            String fileType, double restartFrequency, double temperature, boolean initVelocities, File dyn) {
        this.targetTemperature = temperature;
        this.dt = timeStep;
        this.printFrequency = (int) printInterval;
        this.restartFile = dyn;
        this.initVelocities = initVelocities;

        /**
         * Convert the print interval to a print frequency.
         */
        printFrequency = 100;
        if (printInterval >= this.dt) {
            printFrequency = (int) (printInterval / this.dt);
        }

        /**
         * Convert save interval to a save frequency.
         */
        saveSnapshotFrequency = 1000;
        if (saveInterval >= this.dt) {
            saveSnapshotFrequency = (int) (saveInterval / this.dt);
        }

        done = false;

        assemblies.stream().parallel().forEach((ainfo) -> {
            MolecularAssembly mola = ainfo.getAssembly();
            CompositeConfiguration aprops = ainfo.props;
            File file = mola.getFile();
            String filename = FilenameUtils.removeExtension(file.getAbsolutePath());
            File archFile = ainfo.archiveFile;
            if (archFile == null) {
                archFile = new File(filename + ".arc");
                ainfo.archiveFile = XYZFilter.version(archFile);
            }
            if (ainfo.pdbFile == null) {
                String extName = FilenameUtils.getExtension(file.getName());
                if (extName.toLowerCase().startsWith("pdb")) {
                    ainfo.pdbFile = file;
                } else {
                    ainfo.pdbFile = new File(filename + ".pdb");
                }
            }
            if (ainfo.xyzFilter == null) {
                ainfo.xyzFilter = new XYZFilter(file, mola, mola.getForceField(), aprops);
            }
            if (ainfo.pdbFilter == null) {
                ainfo.pdbFilter = new PDBFilter(ainfo.pdbFile, mola, mola.getForceField(), aprops);
            }
        });

        String firstFileName = FilenameUtils.removeExtension(molecularAssembly.getFile().getAbsolutePath());

        if (dyn == null) {
            restartFile = new File(firstFileName + ".dyn");
            loadRestart = false;
        } else {
            restartFile = dyn;
            loadRestart = true;
        }

        if (dynFilter == null) {
            dynFilter = new DYNFilter(molecularAssembly.getName());
        }

        dof = forceFieldEnergyOpenMM.calculateDegreesOfFreedom();

        if (!initialized) {
            if (loadRestart) {
                Crystal crystal = molecularAssembly.getCrystal();
                // possibly add check to see if OpenMM supports this space group.
                if (!dynFilter.readDYN(restartFile, crystal, x, v, a, aPrevious)) {
                    String message = " Could not load the restart file - dynamics terminated.";
                    logger.log(Level.WARNING, message);
                    done = true;
                } else {
                    //molecularAssembly.getPotentialEnergy().setCrystal(crystal);
                    forceFieldEnergyOpenMM.setCrystal(crystal);

                    // Load positions into the main FFX data structure, move into primary unit cell, then load to OpenMM.
                    Atom[] atoms = molecularAssembly.getAtomArray();
                    double[] xyz = new double[3];
                    for (int i = 0; i < atoms.length; i++) {
                        int i3 = i * 3;
                        for (int j = 0; j < 3; j++) {
                            xyz[j] = x[i3 + j];
                        }
                        atoms[i].setXYZ(xyz);
                    }
                    molecularAssembly.moveAllIntoUnitCell();
                    forceFieldEnergyOpenMM.loadFFXPositionToOpenMM();

                    forceFieldEnergyOpenMM.setOpenMMVelocities(v, numberOfVariables);
                }
            } else {
                forceFieldEnergyOpenMM.loadFFXPositionToOpenMM();
                if (initVelocities) {
                    getThermostat().setTargetTemperature(targetTemperature);
                    getThermostat().maxwell();
                    forceFieldEnergyOpenMM.setOpenMMVelocities(v, numberOfVariables);
                    //int randomNumber = random.nextInt();
                    //OpenMM_Context_setVelocitiesToTemperature(context, temperature, randomNumber);
                }
            }
        }

        int i = 0;
        running = false;

        // logger.info(" Calling OpenMM Update from MD Init.");
        openMM_Update(i, running);
    }

    /**
     * Start sets up context, write out file name, restart file name, sets the
     * integrator and determines whether the simulation is starting out from a
     * previous molecular dynamics run (.dyn) or if the initial velocities are
     * determined by a Maxwell Boltzmann distribution. This method then calls
     * methods openMMUpdate and takeSteps to run the molecular dynamics
     * simulation.
     *
     * @param numSteps
     */
    @Override
    public void dynamic(int numSteps, double timeStep, double printInterval, double saveInterval, double temperature, boolean initVelocities, File dyn) {
        init(numSteps, timeStep, printInterval, saveInterval, fileType, restartFrequency, temperature, initVelocities, dyn);

        //initialized = true;
        storeState();
        if (intervalSteps == 0 || intervalSteps > numSteps) {
            intervalSteps = numSteps;
        }
        running = true;

        int i = 0;
        time = -System.nanoTime();
        while (i < numSteps) {
            // logger.info(" Calling OpenMM Update prior to MD Steps.");
            openMM_Update(i, running);
            //time = -System.nanoTime();
            takeSteps(intervalSteps);
            //time += System.nanoTime();
            i += intervalSteps;
        }
        // logger.info(" Calling OpenMM Update after MD Steps.");
        openMM_Update(i, running);
        logger.info("");
    }

    public final void integratorToString(IntegratorEnum integrator) {
        if (integrator == null) {
            integratorString = "VERLET";
            logger.info(String.format(" No specified integrator, will use Verlet"));
        } else {
            switch (integratorType) {
                case STOCHASTIC:
                    this.integratorString = "LANGEVIN";
                    break;
                case VELOCITYVERLET:
                    this.integratorString = "VERLET";
                    break;
                default:
                    this.integratorString = "VERLET";
                    logger.warning(String.format(" Integrator %s incompatible with "
                            + "OpenMM MD integration; defaulting to %s", integratorType, this.integratorString));
                    break;
            }
        }

        logger.info(String.format(" Created %s integrator", this.integratorString));
    }

    private void thermostatToString() {
        if (integratorType != null && integratorType == IntegratorEnum.STOCHASTIC) {
            logger.fine(" Ignoring requested thermostat due to Langevin dynamics.");
            thermostatString = "NVE";
            return;
        }

        if (thermostatType == null) {
            thermostatString = "ANDERSEN";
            logger.info(String.format(" No specified thermostat, will use %s", thermostatString));
        } else {
            switch (thermostatType) {
                case ADIABATIC:
                    logger.info(" Adiabatic thermostat specified; will use NVE dynamics");
                    thermostatString = "NVE";
                    break;
                default:
                    thermostatString = "ANDERSEN";
                    logger.info(String.format(" Thermostat %s requested, but incompatible with OpenMM: will use %s thermostat", thermostatType, thermostatString));
                    break;
            }
        }
    }

    public void setIntervalSteps(int intervalSteps) {
        this.intervalSteps = intervalSteps;
        logger.info(String.format(" Interval Steps set at %d", intervalSteps));
    }

    public final void updateIntegrator() {
        forceFieldEnergyOpenMM.setIntegrator(integratorString, dt, targetTemperature);
        integrator = forceFieldEnergyOpenMM.getIntegrator();
        context = forceFieldEnergyOpenMM.getContext();
    }

    private void updateThermostat() {
        switch (thermostatString) {
            case "NVE":
                break;
            case "ANDERSEN":
            default:
                forceFieldEnergyOpenMM.addAndersenThermostat(targetTemperature, collisionFreq);
                break;
        }
    }

    /**
     * Returns the OpenMM DynamicsEngine
     *
     * @return OPENMM
     */
    @Override
    public DynamicsEngine getEngine() {
        return DynamicsEngine.OPENMM;
    }
    
    /**
     * Returns time spent calculating the molecular dynamics trajectory on the GPU
     * @return 
     */
    @Override
    public long getMDTime(){
        return mdTime;
    }
    
    @Override
    public double getStartingTotalEnergy(){
        return startingTotalEnergy;
    }
    
    @Override
    public double getEndTotalEnergy(){
        return endTotalEnergy;
    }
}
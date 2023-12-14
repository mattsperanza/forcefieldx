
package ffx.potential.openmm;

import static edu.uiowa.jopenmm.OpenMMLibrary.OpenMM_AndersenThermostat_create;
import static edu.uiowa.jopenmm.OpenMMLibrary.OpenMM_AndersenThermostat_destroy;
import static edu.uiowa.jopenmm.OpenMMLibrary.OpenMM_AndersenThermostat_setDefaultTemperature;
import static edu.uiowa.jopenmm.OpenMMLibrary.OpenMM_AndersenThermostat_setDefaultCollisionFrequency;

/**
 * OpenMM AndersenThermostat.
 */
public class OpenMMAndersenThermostat extends OpenMMForce {

  /**
   * OpenMM AndersenThermostat constructor.
   *
   * @param temperature The temperature.
   * @param frequency   The collision frequency.
   */
  public OpenMMAndersenThermostat(double temperature, double frequency) {
    pointer = OpenMM_AndersenThermostat_create(temperature, frequency);
  }

  /**
   * Set the default temperature.
   *
   * @param temperature The temperature.
   */
  public void setDefaultTemperature(double temperature) {
    OpenMM_AndersenThermostat_setDefaultTemperature(pointer, temperature);
  }

  /**
   * Set the default collision frequency.
   *
   * @param frequency The collision frequency.
   */
  public void setDefaultCollisionFrequency(double frequency) {
    OpenMM_AndersenThermostat_setDefaultCollisionFrequency(pointer, frequency);
  }

  /**
   * Destroy the force.
   */
  public void destroy() {
    if (pointer != null) {
      OpenMM_AndersenThermostat_destroy(pointer);
      pointer = null;
    }
  }
}

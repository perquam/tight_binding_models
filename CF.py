class ConfigurationParameters:
    """Class storing all configuration parameters.

    Attributes
    ----------
    a0_bounds : Tuple[float]
        material lattice constant bounds (in nanometers)  
    V._bounds : Tuple[float]
        bond integrals bounds (in meV)
    e._bounds : Tuple[float]
        tight-binding onsite energies bounds (in meV)
    band_gap : float
        the difference between the maximum energy in valence band
        and the minimum energy in conduciton band (dimensionless)
    frac_of_band_gaps
        the fraction of band structures having band gap,
        assured by the band_gap parameter
    """
    a0_bounds = (0.319, 0.319)
    Vd2sigma_bounds = (-1600.0, 1600.0)
    Vd2pi_bounds = (-1600.0, 1600.0)
    Vd2delta_bounds = (-1600.0, 1600.0)
    e0_bounds = (-2000.0, 2000.0)
    e1_bounds = (-2000.0, 2000.0)
    e2_bounds = (-2000.0, 2000.0)
    band_gap = 0.011
    frac_of_band_gaps = 1.0
    
config_parameters = ConfigurationParameters()

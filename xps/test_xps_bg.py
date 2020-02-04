pathRef = '/Users/pabloherrero/sabat/xps_spectra/2019_10_28_Au_crystal_clean/20191028_Au(788)_clean.xy'
path =  '/Users/pabloherrero/sabat/xps_spectra/2019_10_31_Au_crystal_sputter_2kev/20191031_FBI_Ba_Au(788)_2ndSputter_2kev.xy'
xp = xps_data_import(path=path)
xpRef = xps_data_import(path=pathRef)

def test_scale_and_plot_spectra():
    region = 'overview'
    y_sc, scale_av, indmax = scale_and_plot_spectra(xp = xp, xpRef=xpRef, region=region, lb=('xp', 'ref'))
    ratio_scaled_peaks = xpRef.dfx[region].counts[indmax] / y_sc[indmax]

    assert round(ratio_scaled_peaks) == 1, "Peak heights do not coincide"

def test_normalise_dfx():
    region = 'overview'
    y_sc, scale_av, indmax = scale_and_plot_spectra(xp = xp, xpRef=xpRef, region=region, lb=('xp', 'ref'))
    xp_norm = normalise_dfx(xp, indmax)
    y_norm = xp_norm.dfx['overview'].counts
    assert np.max(y_norm) == 1,  "Incorrect normalization"

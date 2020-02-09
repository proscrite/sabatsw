pathRef = '/Users/pabloherrero/sabat/xps_spectra/2019_10_28_Au_crystal_clean/20191028_Au(788)_clean.xy'
path =  '/Users/pabloherrero/sabat/xps_spectra/2019_10_31_Au_crystal_sputter_2kev/20191031_FBI_Ba_Au(788)_2ndSputter_2kev.xy'
xp = xps_data_import(path=path)
xpRef = xps_data_import(path=pathRef)

def test_scale_and_plot_spectra():
    """If peaks are shifted wrt each other, then ratio_scaled_peaks is not exactly 1 but should be close enough"""
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

def test_bulk_bg_subtract():
    regions = ['In3d5/2', 'Sn3d5/2', 'C1s', 'O1s']
    bg_exps = bulk_bg_subtract([xp], regions)  # Perform bg subtraction on several regions
    xp_test = bulk_bg_subtract(bg_exps, regions)   # Repeat

    no_bg = []          # Check whether no effect took place for some region
    for r in regions:
         no_bg.append((xp_test[0].dfx[r].counts == bg_exps[0].dfx[r].counts).any())
    assert np.array(no_bg).any()

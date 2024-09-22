##Data Munging


import numpy as np
import matplotlib.pyplot as plt
from TheCannon import apogee
tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("example_DR10/Data")
tr_label = apogee.load_labels("example_DR10/reference_labels.csv")
# print(tr_ID.shape)
# print(wl.shape)
# print(tr_flux.shape)
# print(tr_label.shape)
index = np.where(tr_ID=='2M21332216-0048247')[0][0]
flux = tr_flux[index]
# plt.plot(wl, flux, c='k')
# plt.show()
ivar = tr_ivar[index]
choose = ivar > 0
# plt.plot(wl[choose], flux[choose], c='k')
# plt.show()

test_ID = tr_ID
test_flux = tr_flux
test_ivar = tr_ivar

from TheCannon import dataset
ds = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

# print(ds.tr_ID)
# print(ds.tr_flux)
# print(ds.wl)

ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])
fig = ds.diagnostics_SNR()
plt.savefig("result.png")
# fig = ds.diagnostics_ref_labels()
# plt.savefig("result2.png")

# ds.ranges = [[371,3192], [3697,5997], [6461,8255]]
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)
# contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
ds.set_continuum(contmask)
cont = ds.fit_continuum(3, "sinusoid")
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)
plt.plot(wl, norm_tr_flux[10,:])
ds.tr_flux = norm_tr_flux
ds.tr_ivar = norm_tr_ivar
ds.test_flux = norm_test_flux
ds.test_ivar = norm_test_ivar

################################################################
#########################Runnning the Cannon####################
################################################################

from TheCannon import model
md = model.CannonModel(2, useErrors=False)
md.fit(ds)
md.diagnostics_contpix(ds)
#md.diagnostics_leading_coeffs(ds)
md.diagnostics_plot_chisq(ds)

label_errs = md.infer_labels(ds)
test_labels = ds.test_label_vals
# ds.diagnostics_test_step_flagstars()
# ds.diagnostics_survey_labels()
ds.diagnostics_1to1()

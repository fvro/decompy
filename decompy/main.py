import copy
import dill
import scipy
import dynesty
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os.path
from getdist import plots, MCSamples
from skimage import measure, filters
from scipy import ndimage
from scipy import special
from scipy.integrate import trapz
from scipy.stats import uniform
from scipy.stats import norm
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.modeling.models import Sersic2D
from astropy.modeling.models import Ellipse2D
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from astropy.modeling.functional_models import Gaussian2D
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm
from matplotlib.colors import PowerNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
from dataclasses import dataclass
from reproject import reproject_interp
from dynesty import plotting as dyplot
from dynesty.pool import Pool
import dynesty.utils

dynesty.utils.pickle_module = dill


G = const.G.to(u.kpc*(u.km/u.s)**2/u.M_sun) # G = G.value

uprior = ['u','uni','unif','uniform']
ulprior = ['ul','unilog','uniflog','uniformlog']
gprior = ['g','gauss','gaussian']
lnprior = ['ln','lognorm','lognormal']
tprior = ['t', 'trunc','truncated']




@dataclass
class Galaxy_Data:
	ngal: int
	ID: int
	name: str
	redshift: float
	phys_scale: float
	gal_group: str

	Mstar: float
	Mstar_err: float
	L_co: float
	L_co_err: float
	rl: float
	SF: str

	radii_kpc: float
	radii_arcsec: float
	vrot: float
	err_vrot: float
	vdisp: float
	err_vdisp: float

	mom0gas: float
	gas_px_scale: float
	rms_gas: float
	threshold_value_gas: float
	ALMA_header: float

	HST_filter: str
	HST_px_scale: float
	HST_data: float
	HST_header: float
	rms_HST: float
	threshold_value_HST: float

	JWST_flag: int
	JWST_filter: str
	JWST_px_scale: float
	JWST_data: float
	JWST_header: float
	rms_JWST: float
	threshold_value_JWST: float




	def plot_data(self, fig):

		rows = 4
		cols = 3
		gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05, top=0.99, bottom=0.01, right=0.99, left=0.01)


		wmap_HST = WCS(self.HST_header, naxis=2)
		ALMA_size = self.mom0gas.shape[0]


		cutsize = [100, 45, 45, 40, 35, 50, 25, 35, 30, 40, 40]
		Norm    = [0.4, 0.7, 0.5, 0.65, 0.6, 0.5, 0.6, 2.5, 0.9, 2.5, 0.6]


		gcutsize = cutsize[self.ngal]
		gnorm    = Norm[self.ngal]

		if(self.JWST_flag == 1):
			wmap_JWST = WCS(self.JWST_header, naxis=2)

			JWST_cut, ALMA_cut, wcs, ext = cut_two_images(image1=self.JWST_data, header1=self.JWST_header, image2=self.mom0gas, header2=self.ALMA_header, cut_size = gcutsize)

			ax = fig.add_subplot(gs[(self.ngal)//cols, (self.ngal)%cols], projection=wcs)
			ax.imshow(JWST_cut, origin='lower', cmap = 'viridis', norm=PowerNorm(gnorm), extent=ext)


			filter = self.JWST_filter

		else:
			wmap_ALMA = WCS(self.ALMA_header, naxis=2)

			HST_cut, ALMA_cut, wcs, ext = cut_two_images(image1=self.HST_data, header1=self.HST_header, image2=self.mom0gas, header2=self.ALMA_header, cut_size = gcutsize)

			ax = fig.add_subplot(gs[(self.ngal)//cols, (self.ngal)%cols], projection=wcs)
			ax.imshow(HST_cut, origin='lower', cmap = 'viridis', norm=PowerNorm(gnorm), extent=ext)
			filter = self.HST_filter


		lon = ax.coords[0]
		lat = ax.coords[1]
		lon.set_ticks_visible(False)
		lon.set_ticklabel_visible(False)
		lat.set_ticks_visible(False)
		lat.set_ticklabel_visible(False)
		lon.set_axislabel('')
		lat.set_axislabel('')


		levels = np.asarray([2, 4, 8, 16])*self.rms_gas
		ax.contour(ALMA_cut, levels=levels, colors='cyan', extent=ext)
		levels = np.asarray([-4, -2])*self.rms_gas
		ax.contour(ALMA_cut, levels=levels, colors='teal', linestyles='dashed', extent=ext)

		kpc = 1/(self.gas_px_scale*(1/cosmo.arcsec_per_kpc_proper(self.redshift).value))
		size_x = ext[1]-ext[0]
		kpc_ext = kpc/size_x
		stepx = (ext[3]-ext[2])*0.1
		stepy = (ext[1]-ext[0])*0.08

		axz = fig.add_subplot(gs[(self.ngal)//cols, (self.ngal)%cols])
		axz.axis('off')


		conv = Angle(np.abs(self.ALMA_header['CDELT1']), u.deg).arcsec
		majb = Angle(np.abs(self.ALMA_header['BMAJ']), u.deg).arcsec/(2*conv) #arcsec
		minb = Angle(np.abs(self.ALMA_header['BMIN']), u.deg).arcsec/(2*conv)
		pab = self.ALMA_header['BPA']
		posa = np.radians(pab-90)  
		t = np.linspace(0,2*np.pi,100)
		xt = ext[0]+1.6*stepx+majb*np.cos(posa)*np.cos(t)-minb*np.sin(posa)*np.sin(t)  
		yt = ext[2]+1.7*stepy+majb*np.sin(posa)*np.cos(t)+minb*np.cos(posa)*np.sin(t)
		ax.plot(xt, yt, '-', c='white', lw=3, zorder=9)
		#ax.fill_between(xt, yt, hatch='////', fc='white', zorder=8)


		ax.plot((ext[1]-stepx-kpc/2, ext[1]-stepx+kpc/2), (ext[2]+stepy, ext[2]+stepy), 'white')
		axz.text(0.95, 0.01, '1 kpc', fontsize=8, c='white', horizontalalignment='right', verticalalignment='bottom', fontweight='semibold')

		axz.text(0.05, 0.98, f'ID{self.ID}', c='white', fontsize=11, horizontalalignment='left', verticalalignment='top', fontweight='semibold')
		axz.text(0.05, 0.90, f'z={self.redshift}', c='white', fontsize=9, horizontalalignment='left', verticalalignment='top', fontweight='semibold')
		axz.text(0.95, 0.98, f'{filter}', c='white', fontsize=9, horizontalalignment='right', verticalalignment='top', fontweight='semibold')


		if(self.ngal > 5):
			for axis in ['top','bottom','left','right']:
				ax.spines[axis].set_linewidth(50)





class ADC:
	def __init__(self, galID, phys_scale, radii, Rgas, err_Rgas, Vrot, err_Vrot, vdisp, err_vdisp, ngal=None, path=None):
		self.nrandom    = 1000
		self.galID      = galID
		if(ngal != None): self.ngal = ngal
		self.phys_scale = phys_scale
		self.Vrot       = Vrot
		self.err_Vrot   = err_Vrot
		self.radii      = radii
		self.Rgas       = make_random_gaussian(Rgas, err_Rgas, self.nrandom)
		self.vrot       = make_random_gaussian(Vrot, err_Vrot, self.nrandom)
		self.vdisp      = make_random_gaussian(vdisp, err_vdisp, self.nrandom)
		self.path       = path


	def run(self):
		#calculate the derivative of ln sigma:
		slope = ddR_lnsigma(self.vdisp, self.radii, self.nrandom)

		#calculate the pressure support per radii:
		Va_square = VA_square(self.radii, self.vdisp, slope, self.Rgas, self.nrandom)

		#calculate circular velocity per radii:
		Vc = np.sqrt(self.vrot**2+Va_square)

		vcirc     = np.zeros(len(self.radii))
		err_vcirc = np.zeros(len(self.radii))


		for r, rad in enumerate(self.radii):
			vcirc[r]     = np.nanpercentile(Vc[r], 50)
			err_vcirc[r] = np.nanstd(Vc[r])
			#print('Vc (r =', r, 'kpc):', vcirc[r], '+-', err_vcirc[r])
		self.vcirc     = vcirc
		self.err_vcirc = err_vcirc


		if(self.path == None):
			with open('./output/adc/'+str(self.galID)+'_vcirc.txt', 'w') as f:
				f.write("#R vcirc err_vcirc\n")
				for val1, val2, val3 in zip(self.radii, self.vcirc, self.err_vcirc):
					f.write(f"{val1} {val2} {val3}\n")
		else:
			with open(self.path+''+str(self.galID)+'_vcirc.txt', 'w') as f:
				f.write("#R vcirc err_vcirc\n")
				for val1, val2, val3 in zip(self.radii, self.vcirc, self.err_vcirc):
					f.write(f"{val1} {val2} {val3}\n")		


	def plot_adc(self, fig):

		rows = 4
		cols = 3
		gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.13, wspace=0.2, top=0.99, bottom=0.05, right=0.98, left=0.08)

		colours = ['#C2185B', '#5C6BC0']

		ax = fig.add_subplot(gs[(self.ngal)//cols, (self.ngal)%cols])

		ax.plot(self.radii*self.phys_scale, self.Vrot, color=colours[0], lw=2, linestyle='solid', label = '$V_\mathrm{rot}$', alpha=0.85, zorder=8)
		#ax.errorbar(self.radii*self.phys_scale, self.Vrot, color='gray', lw= 2, linestyle='solid', yerr=self.err_Vrot)
		ax.fill_between(self.radii*self.phys_scale, self.Vrot + self.err_Vrot, self.Vrot, color=colours[0], alpha=0.5)
		ax.fill_between(self.radii*self.phys_scale, self.Vrot - self.err_Vrot, self.Vrot, color=colours[0], alpha=0.5)

		ax.plot(self.radii*self.phys_scale, self.vcirc, color=colours[1], lw=2, linestyle='-.', label = '$V_\mathrm{circ}$', alpha=1, zorder=10)
		ax.errorbar(self.radii*self.phys_scale, self.vcirc, color=colours[1], lw= 2, linestyle='-.', yerr=self.err_vcirc, alpha=1, zorder=10)


		xmax = self.radii[-1]*self.phys_scale*1.15
		ymax = np.max(self.vcirc)*1.25

		ax.text(xmax*0.05, ymax*0.1, f'ID{self.galID}', c='k', fontsize=11, horizontalalignment='left', verticalalignment='top', fontweight='semibold')

		ax.set_ylim(0, ymax)
		ax.set_xlim(0, xmax)

		step_x = np.round(xmax/2)/2
		init_x = np.round(xmax/2)/2
		xticks = np.arange(init_x, xmax, step_x)

		step_y = 50 * round((ymax/4)/50)
		init_y = 50 * round((ymax/4)/50)
		yticks = np.arange(init_y, ymax, step_y)


		ax.set_xticks(xticks)
		ax.set_yticks(yticks)

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())

		ax.tick_params(axis='both', which='both', width=1, direction='in', labelsize=10)


		if((self.ngal)%cols == 0 ): ax.set_ylabel('V (km/s)')
		if((self.ngal) > 8 ): ax.set_xlabel('R (kpc)')

		if(self.ngal == 10):
			plt.legend(bbox_to_anchor=(1.75, 0.85), fontsize=14, fancybox = True)



class dySB:
	def __init__(self, model, data, header, rms, name, redshift, multicomponent=False, threshold=None, doublecomponent=False, type_double=None):
		pars		= model['pars'].copy()
		self.flags      = model['flags'].copy()
		self.data       = data
		self.header     = header
		self.rms        = rms

		self.multicomponent = multicomponent
		self.doublecomponent = doublecomponent
		self.type_double = type_double

		self.redshift   = redshift
		self.x, self.y  = np.meshgrid(np.arange(self.data.shape[-1]), np.arange(self.data.shape[-2]))
		self.phys_scale = 1/cosmo.arcsec_per_kpc_proper(self.redshift).value #kpc/arcsec
		self.inv_sigma2 = 1./(self.rms**2)


		self.mask = np.ones((data.shape[-1], data.shape[-2]), dtype=bool)
		self.data_m     = self.data[self.mask]
		x_m             = self.x[self.mask]
		y_m             = self.y[self.mask]


		if(self.flags['datatype'] == 'gas'):
			self.name       = name+'_gas'
			self.beam       = self.header['BMIN']*3600. #in arcsec
			self.kernel     = Gaussian2DKernel(x_stddev=self.header['BMAJ']/(self.header['CDELT2']*2.355), y_stddev=self.header['BMIN']/(self.header['CDELT2']*2.355), theta=np.deg2rad(90.0-self.header['BPA'])) #in deg
			self.px_scale = self.header['CDELT2']*3600. #arcsec/px



		if(self.flags['datatype'] == 'stars'):
			self.name       = name+'_stars'
			self.beam       = self.header['PSFMIN'] #in arcsec
			self.kernel     = Gaussian2DKernel(x_stddev = (self.header['PSFMAJ']/(3600*self.header['CD2_2']*2.355)),  y_stddev = (self.header['PSFMIN']/(3600*self.header['CD2_2']*2.355)), theta    = (np.deg2rad(90.0-self.header['PSFPA'])) ) #in deg
			self.bmaj       = (self.header['PSFMAJ']/(3600*self.header['CD2_2']*2.355))
			self.bmin       = (self.header['PSFMIN']/(3600*self.header['CD2_2']*2.355))
			self.theta      = (np.deg2rad(90.0-self.header['PSFPA']))
			self.px_scale = self.header['CD2_2']*3600. #arcsec/px


		keys = ['I$_{e}$',r'r$_{eff}$','x0','y0','n','ϵ', 'PA']



		if(multicomponent == False):
			print(f'Fitting one component: {self.name}')
			self.Ncomp = 1
			self.pars = pars
			self.keys = keys
		else:
			print(f'Fitting multiple components: {self.name}')
			sources = source_finder(image=self.data, threshold_value=threshold, name=self.name)
			self.Ncomp = sources.shape[0]
			coord_sources = sources
			print(f'Number of components: {self.Ncomp}')

			Npars = duplicate_dict(pars, self.Ncomp)
			self.pars = Npars
			Nkeys = duplicate_list(keys, self.Ncomp)
			self.keys = Nkeys
			for pp in range(0, self.Ncomp):
				idx = pp+1
				self.pars[f'x0,{idx}']['prior']['low'] = coord_sources[pp][0]-10
				self.pars[f'x0,{idx}']['prior']['up']  = coord_sources[pp][0]+10
				self.pars[f'y0,{idx}']['prior']['low'] = coord_sources[pp][1]-10
				self.pars[f'y0,{idx}']['prior']['up']  = coord_sources[pp][1]+10

		if(name == 'ID13'):
			self.Ncomp = 3
			Npars = duplicate_dict(pars, self.Ncomp)
			self.pars = Npars
			Nkeys = duplicate_list(keys, self.Ncomp)
			self.keys = Nkeys

			coords_x = [58, 30, 72]
			coords_y = [59, 70, 39]

			for pp in range(0, self.Ncomp):
				idx = pp+1
				self.pars[f'x0,{idx}']['prior']['low'] = coords_x[pp]-10
				self.pars[f'x0,{idx}']['prior']['up']  = coords_x[pp]+10
				self.pars[f'y0,{idx}']['prior']['low'] = coords_y[pp]-10
				self.pars[f'y0,{idx}']['prior']['up']  = coords_y[pp]+10

			Fig, Ax = plt.subplots()
			Ax.imshow(self.data, origin='lower', norm=PowerNorm(0.2))  # Display the image
			source_radius = self.data.shape[0]*0.1
			for j in range(0, 3):
				if(j == 0):
					lw = 2.5
					color = 'red'
				else:
					lw = 1.5
					color = 'yellow'
				circle = Circle((coords_x[j], coords_y[j]), source_radius, edgecolor=color, linewidth=lw, facecolor='none')
				Ax.add_patch(circle)

			plt.savefig('./output/figures/others/ID13_stars_sources.png')




		if(self.doublecomponent):
			print(f'Fitting a double component: {self.name}')
			if(self.type_double == 'exp_psf'):
				self.name = self.name+'_'+self.type_double
				self.keys = self.keys
				self.keys.append('Amp')
				self.pars['Amp'] = {'value': 0.,  'vary': True,	'prior': {'type': 'uni',  'low': self.rms*1.5,	'up': 1.5*np.amax(self.data)}}
				if(self.Ncomp > 1):
					self.pars['n,1']['vary'] = False
					self.pars['n,1']['value'] = 1.0
				else:
					self.pars['n']['vary'] = False
					self.pars['n']['value'] = 1.0

			elif(self.type_double == 'sersic_psf'):
				self.name = self.name+'_'+self.type_double
				self.keys = self.keys
				self.keys.append('Amp')
				self.pars['Amp'] = {'value': 0.,  'vary': True,	'prior': {'type': 'uni',  'low': self.rms*1.5,	'up': 1.5*np.amax(self.data)}}
				if(self.Ncomp > 1):
					self.pars['n,1']['vary'] = True
				else:
					self.pars['n']['vary'] = True

			elif(self.type_double == 'exp_sersic'):
				self.name = self.name+'_'+self.type_double
				self.keys = self.keys
				self.keys.append('n_ser')
				self.keys.append('Ieff_ser')
				self.keys.append('Reff_ser')
				self.keys.append('eps_ser')
				self.keys.append('PA_ser')

				self.pars['n_ser'] = {'value': 1.0, 'vary': True,'prior': {'type': 'uni', 'low': 0.2,'up': 10.0}}
				self.pars['Ieff_ser'] = {'value': 0.,  'vary': True,'prior': {'type': 'uni', 'low': rms*1.5,'up': 1.5*np.amax(data)}}
				self.pars['Reff_ser'] = {'value': 0.8, 'vary': True,'prior': {'type': 'uni', 'low': 2.,	'up': 50.}}
				self.pars['eps_ser'] = {'value': 3.4, 'vary': True,'prior': {'type': 'uni',  'low': 0.01,'up': 1.0}}
				self.pars['PA_ser'] = {'value': 3.4, 'vary': True,'prior': {'type': 'uni',  'low': 0.0,'up': 3.14}}

				if(self.Ncomp > 1):
					self.pars['n,1']['vary'] = False
					self.pars['n,1']['value'] = 1.0
				else:
					self.pars['n']['vary'] = False
					self.pars['n']['value'] = 1.0


			"""
				if(self.type_double == 'exp_bulge'):
					self.pars['n,1']['vary'] = False
					self.pars['n,1']['value'] = 1.0
					self.pars['n,2']['vary'] = False
					self.pars['n,2']['value'] = 4.0
			"""





	def run(self, fit='new'):
	# Project hypercube on prior phase space
	# ----------------------------------------------------------------------
		def ediprior(x):
			xind = 0
			xppf = []
			for par in self.pars:
				if self.pars[par]['vary']:
					if self.pars[par]['prior']['type'].lower() in gprior:
						xppf.append(scipy.stats.norm.ppf(x[xind],loc=self.pars[par]['prior']['loc'],scale=self.pars[par]['prior']['scale']))
					elif self.pars[par]['prior']['type'].lower() in uprior:
						xppf.append(scipy.stats.uniform.ppf(x[xind],loc=self.pars[par]['prior']['low'],scale=self.pars[par]['prior']['up']-self.pars[par]['prior']['low']))
					elif self.pars[par]['prior']['type'].lower() in ulprior:
						xppf.append(10.0**scipy.stats.uniform.ppf(x[xind],loc=np.log10(self.pars[par]['prior']['low']),scale=np.log10(self.pars[par]['prior']['up'])-np.log10(self.pars[par]['prior']['low'])))
					xind += 1
			return np.array(xppf)

	# Compute log-likelihood
	# ----------------------------------------------------------------------  
		def logprob(x):

			xind = 0
			pmod = copy.deepcopy(self.pars)
			for par in self.pars: 
				if self.pars[par]['vary']:
					pmod[par]['value'] = x[xind]
					xind += 1

			if self.flags['profile'] == 'sersic':

				if(self.multicomponent == False):
					model = sersic_fit(pmod['I$_{e}$']['value'], pmod[r'r$_{eff}$']['value'], pmod['x0']['value'], pmod['y0']['value'], pmod['n']['value'], pmod['ϵ']['value'], pmod['PA']['value'], self.x, self.y, self.kernel, self.mask)

					if(self.doublecomponent):
						if(self.type_double == 'exp_psf' or self.type_double == 'sersic_psf'):
							psf_model = psf_fit(pmod['Amp']['value'], pmod['x0']['value'], pmod['y0']['value'], self.bmaj, self.bmin, self.theta, self.x, self.y, self.mask)
							model = sersic_fit(pmod['I$_{e}$']['value'], pmod[r'r$_{eff}$']['value'], pmod['x0']['value'], pmod['y0']['value'], pmod['n']['value'], pmod['ϵ']['value'], pmod['PA']['value'], self.x, self.y, self.kernel, self.mask) + psf_model
						elif(self.type_double == 'exp_sersic'):
							exp_model = sersic_fit(pmod['I$_{e}$']['value'], pmod[r'r$_{eff}$']['value'], pmod['x0']['value'], pmod['y0']['value'], pmod['n']['value'], pmod['ϵ']['value'], pmod['PA']['value'], self.x, self.y, self.kernel, self.mask)
							sersic_model = sersic_fit(pmod['Ieff_ser']['value'], pmod['Reff_ser']['value'], pmod['x0']['value'], pmod['y0']['value'], pmod['n_ser']['value'], pmod['eps_ser']['value'], pmod['PA_ser']['value'], self.x, self.y, self.kernel, self.mask)
							model = exp_model + sersic_model


						#elif(self.type_double == 'exp_sersic' or self.type_double == 'exp_bulge'):
							#print("Error: exp_sersic not implemented yet")
							#break
							#model = sersic_fit(pmod['I$_{e}$,1']['value'], pmod[r'r$_{eff}$,1']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,1']['value'], pmod['ϵ,1']['value'], pmod['PA,1']['value'], self.x, self.y, self.kernel, self.mask) + sersic_fit(pmod['I$_{e}$,2']['value'], pmod[r'r$_{eff}$,2']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,2']['value'], pmod['ϵ,2']['value'], pmod['PA,2']['value'], self.x, self.y, self.kernel, self.mask)

				else:
					model = fit_sersic_components(pmod, self)

					if(self.doublecomponent):
						if(self.type_double == 'exp_psf' or self.type_double == 'sersic_psf'):
							psf_model = psf_fit(pmod['Amp']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], self.bmaj, self.bmin, self.theta, self.x, self.y, self.mask)
							model = fit_sersic_components(pmod, self) + psf_model

						elif(self.type_double == 'exp_sersic'):
							exp_model = sersic_fit(pmod['I$_{e}$,1']['value'], pmod[r'r$_{eff}$,1']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,1']['value'], pmod['ϵ,1']['value'], pmod['PA,1']['value'], self.x, self.y, self.kernel, self.mask)
							sersic_model = sersic_fit(pmod['Ieff_ser']['value'], pmod['Reff_ser']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n_ser']['value'], pmod['eps_ser']['value'], pmod['PA_ser']['value'], self.x, self.y, self.kernel, self.mask)
							model = exp_model + sersic_model

						#elif(self.type_double == 'exp_sersic' or self.type_double == 'exp_bulge'):
							#print("Error: exp_sersic not implemented yet")
							#break
							#model = sersic_fit(pmod['I$_{e}$,1']['value'], pmod[r'r$_{eff}$,1']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,1']['value'], pmod['ϵ,1']['value'], pmod['PA,1']['value'], self.x, self.y, self.kernel, self.mask) + sersic_fit(pmod['I$_{e}$,2']['value'], pmod[r'r$_{eff}$,2']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,2']['value'], pmod['ϵ,2']['value'], pmod['PA,2']['value'], self.x, self.y, self.kernel, self.mask)


			return -0.5*(np.sum((self.data_m-model)**2 * self.inv_sigma2))


		if(fit == 'new'):
			print('The fit is set to \'new\', initiating a new fit.')
			self.ndim = np.count_nonzero(np.array([self.pars[par]['vary'] for par in self.keys]))

			with Pool(8, logprob, ediprior) as pool:
				self.sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim=self.ndim, nlive=10*self.ndim, pool=pool)
				self.sampler.run_nested(checkpoint_file = self.name+'_checkpointing.save')

			self.res1 = self.sampler.results

			namechain = self.name+'_dySB.dill'
			with open(namechain,'wb') as file: dill.dump(self.res1, file) 
			self.wgt1 = np.exp(self.sampler.results['logwt']-self.sampler.results['logz'][-1])

		elif(fit == 'resume'):
			print('The fit is set to \'resume\', resuming the previous fit.')
			if(os.path.exists(self.name+'_checkpointing.save')):
				self.ndim = np.count_nonzero(np.array([self.pars[par]['vary'] for par in self.keys]))
				with Pool(8, logprob, ediprior) as pool:
					self.sampler = dynesty.NestedSampler.restore(self.name+'_checkpointing.save')
					self.sampler.run_nested(resume=True)
				self.res1 = self.sampler.results
				namechain = self.name+'_dySB.dill'
				with open(namechain,'wb') as file: dill.dump(self.res1, file) 
				self.wgt1 = np.exp(self.sampler.results['logwt']-self.sampler.results['logz'][-1])
			else: print('Checkpointing file does not exist. Please select a different argument for fitting.\nThe options are \'new\' for a new run, \'resume\' for continuing a previous fit or \'read\' for reading an existing finished posterior chain.')

		elif(fit == 'read'):
			print('The fit is set to \'read\', reading the results of a previous fit.')
			namechain = self.name+'_dySB.dill'
			with open(namechain,'rb') as file: 
				self.res1 = dill.load(file)
			self.ndim = self.res1.samples.shape[1]
			#print('number of dimensions:', self.ndim)
			self.wgt1 = np.exp(self.res1['logwt']-self.res1['logz'][-1])
			#print('logz:', self.res1['logz'][-1])

		xind = 0
		self.labels = []
		self.bestfit = {}
		for par in self.pars:
			if self.pars[par]['vary']:
				if(xind < self.ndim):
					self.labels.append(par)
					self.bestfit.update({par: self.res1.samples[:,xind]})
					xind += 1	


		parnames = self.labels
		samples = self.res1.samples
		weights = self.wgt1
		# Save the samples and weights to a text file
		hdr = ' '.join(parnames + ["weights"])
		np.savetxt('./output/dySB/chains/dySB_chains_with_weights_'+self.name+'.txt', np.column_stack((samples, weights)), header=hdr)





	def plot_dySB(self, path=None):

		fig, axes = plt.subplots(self.ndim, self.ndim, figsize=(10,10))

		f, ax = dyplot.cornerplot(self.res1, color='k', truth_color='black', show_titles=True,
						max_n_ticks=3, fig=(fig, axes[:,:self.ndim]), hist_kwargs={'alpha': 0.4},
						title_kwargs={'fontsize' : 12, 'y': 1.02, 'x': 0.52},
						title_fmt='.2f',
						title_quantiles=(0.16, 0.5, 0.84),
						quantiles = (0.16, 0.5, 0.84),
						label_kwargs={'fontsize' : 12},
						truth_kwargs={'fontsize' : 12},
						labels=self.labels)

		plt.suptitle(self.name+'_dySB')

		if(path == None):
			if(self.flags['datatype'] == 'stars'): plt.savefig('./output/dySB/plots/stars/og_cornerplots/'+self.name+'_sb_corner.png', format='png',dpi=300, bbox_inches='tight')
			if(self.flags['datatype'] == 'gas'): plt.savefig('./output/dySB/plots/gas/og_cornerplots/'+self.name+'_sb_corner.png', format='png',dpi=300, bbox_inches='tight')
		else:
			if(self.flags['datatype'] == 'stars'): plt.savefig(path+self.name+'_sb_corner.png', format='png',dpi=300, bbox_inches='tight')
			if(self.flags['datatype'] == 'gas'): plt.savefig(path+self.name+'_sb_corner.png', format='png',dpi=300, bbox_inches='tight')




		self.mean = np.array([dynesty.utils.quantile(self.res1.samples[:,p],[0.50,0.16,0.84],weights=self.wgt1) for p in range(self.res1.samples.shape[1])]).T


		xind = 0
		pmod = copy.deepcopy(self.pars)

		for par in self.pars: 
			if self.pars[par]['vary']: 
				pmod[par]['value'] = self.mean[0][xind]
				xind += 1

		fig=plt.figure(figsize=(16, 5)) 


		grid = gridspec.GridSpec(1,3, left=0.1, bottom=0.1, top=0.94, right=0.99, wspace=0.3) 
		ax0  = plt.subplot(grid[0])
		ax1  = plt.subplot(grid[1])
		ax2  = plt.subplot(grid[2])

		mod = np.zeros_like(self.x)


		if(self.multicomponent):

			Ieff = pmod['I$_{e}$,1']['value']
			reff = pmod[r'r$_{eff}$,1']['value']
			eps  = pmod['ϵ,1']['value']
			n    = pmod['n,1']['value']
			X0   = pmod['x0,1']['value']
			Y0   = pmod['y0,1']['value']
			PA   = pmod['PA,1']['value']

		else:
			Ieff = pmod['I$_{e}$']['value']
			reff = pmod[r'r$_{eff}$']['value']
			eps  = pmod['ϵ']['value']
			n    = pmod['n']['value']
			X0   = pmod['x0']['value']
			Y0   = pmod['y0']['value']
			PA   = pmod['PA']['value']


		if(self.multicomponent == True and self.doublecomponent == False):
			mod = plot_sersic_components(pmod, self)

		if(self.multicomponent == False and self.doublecomponent == False):
			mod = sersic_plot(Ieff, reff, X0, Y0, n, eps, PA, self.x, self.y, self.kernel)

		if(self.multicomponent == False and self.doublecomponent == True):
			if(self.type_double == 'exp_psf' or self.type_double == 'sersic_psf'):
				mod1   = sersic_plot(Ieff, reff, X0, Y0, n, eps, PA, self.x, self.y, self.kernel)
				mod2   = psf_plot(pmod['Amp']['value'], X0, Y0, self.bmaj, self.bmin, self.theta, self.x, self.y)
				mod    = mod1 + mod2
			elif(self.type_double == 'exp_sersic'):
				mod1 = sersic_plot(Ieff, reff, X0, Y0, n, eps, PA, self.x, self.y, self.kernel)
				mod2 = sersic_plot(pmod['Ieff_ser']['value'], pmod['Reff_ser']['value'], X0, Y0, pmod['n_ser']['value'], pmod['eps_ser']['value'], pmod['PA_ser']['value'], self.x, self.y, self.kernel)
				mod = mod1 + mod2


		if(self.multicomponent == True and self.doublecomponent == True):
			if(self.type_double == 'exp_psf' or self.type_double == 'sersic_psf'):
				mod1   = sersic_plot(Ieff, reff, X0, Y0, n, eps, PA, self.x, self.y, self.kernel)
				mod2   = psf_plot(pmod['Amp']['value'], X0, Y0, self.bmaj, self.bmin, self.theta, self.x, self.y)
				mod    = plot_sersic_components(pmod, self) + mod2

			elif(self.type_double == 'exp_sersic'):
				mod1 = sersic_plot(Ieff, reff, X0, Y0, n, eps, PA, self.x, self.y, self.kernel)
				mod2 = sersic_plot(pmod['Ieff_ser']['value'], pmod['Reff_ser']['value'], X0, Y0, pmod['n_ser']['value'], pmod['eps_ser']['value'], pmod['PA_ser']['value'], self.x, self.y, self.kernel)
				mod = plot_sersic_components(pmod, self) + mod2

		"""
		if(self.type_double == 'exp_sersic' or self.type_double == 'exp_bulge'):
			mod1   = sersic_plot(pmod['I$_{e}$,1']['value'], pmod[r'r$_{eff}$,1']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,1']['value'], pmod['ϵ,1']['value'], pmod['PA,1']['value'], self.x, self.y, self.kernel)
			mod2   = sersic_plot(pmod['I$_{e}$,2']['value'], pmod[r'r$_{eff}$,2']['value'], pmod['x0,1']['value'], pmod['y0,1']['value'], pmod['n,2']['value'], pmod['ϵ,2']['value'], pmod['PA,2']['value'], self.x, self.y, self.kernel)
			mod = mod1 + mod2

			if(pmod[r'r$_{eff}$,2']['value'] > pmod[r'r$_{eff}$,1']['value']):
				reff = pmod[r'r$_{eff}$,2']['value']
				eps  = pmod['ϵ,2']['value']
				X0   = pmod['x0,1']['value']
				Y0   = pmod['y0,1']['value']
				PA   = pmod['PA,2']['value']
			else:
				reff = pmod[r'r$_{eff}$,1']['value']
				eps  = pmod['ϵ,1']['value']
				X0   = pmod['x0,1']['value']
				Y0   = pmod['y0,1']['value']
				PA   = pmod['PA,1']['value']
		"""

		if(self.doublecomponent):
			if(path != None):
				fits.writeto(path+self.name+'_model1.fits', mod1, self.header, overwrite=True)
				fits.writeto(path+self.name+'_model2.fits', mod2, self.header, overwrite=True)
			#else:
				#fits.writeto('./output/dySB/models/'+self.name+'_sersic_model1.fits', mod1, self.header, overwrite=True)
				#fits.writeto('./'+self.name+'_sersic_model2.fits', mod2, self.header, overwrite=True)

		if(path == None):

			# Saving model
			if(self.flags['datatype'] == 'stars'):
				fits.writeto('./output/dySB/models/stars/'+self.name+'_model.fits', mod, self.header, overwrite=True)
				data_colour = ['orange', 'blueviolet']

			if(self.flags['datatype'] == 'gas'):
				fits.writeto('./output/dySB/models/gas/'+self.name+'_model.fits', mod, self.header, overwrite=True)
				data_colour = ['turquoise', 'magenta']

		else:
			# Saving model
			if(self.flags['datatype'] == 'stars'):
				fits.writeto(path+self.name+'_model.fits', mod, self.header, overwrite=True)
				data_colour = ['orange', 'blueviolet']

			if(self.flags['datatype'] == 'gas'):
				fits.writeto(path+self.name+'_model.fits', mod, self.header, overwrite=True)
				data_colour = ['turquoise', 'magenta']



		# Plotting maps

		# 1D profiles
		RMAX = reff*self.px_scale*5.0
		INC  = inc(eps)

		if(self.flags['datatype'] == 'stars'): sampling = self.beam/1.5
		if(self.flags['datatype'] == 'gas'): sampling = self.beam/2.


		radii_d, surfb_d, e_surfb_d, rings_d = brightness(data      = self.data,
								RMAX_arcsec = RMAX,
								XPOS        = Y0,
								YPOS        = X0,
								inc0        = INC,
								pa0         = PA,
								pix_to_arc  = self.px_scale,
								FWHM_arcsec = sampling,
								noise       = self.rms)

		radii_m, surfb_m, e_surfb_m, rings_m = brightness(data      = mod,
								RMAX_arcsec = RMAX,
								XPOS        = Y0,
								YPOS        = X0,
								inc0        = INC,
								pa0         = PA,
								pix_to_arc  = self.px_scale,
								FWHM_arcsec = sampling,
								noise       = self.rms)

		if(self.doublecomponent):
			radii_m1, surfb_m1, e_surfb_m1, rings_m1 = brightness(data      = mod1,
										RMAX_arcsec = RMAX,
										XPOS        = Y0,
										YPOS        = X0,
										inc0        = INC,
										pa0         = PA,
										pix_to_arc  = self.px_scale,
										FWHM_arcsec = sampling,
										noise       = self.rms)
			radii_m2, surfb_m2, e_surfb_m2, rings_m2 = brightness(data      = mod2,
										RMAX_arcsec = RMAX,
										XPOS        = Y0,
										YPOS        = X0,
										inc0        = INC,
										pa0         = PA,
										pix_to_arc  = self.px_scale,
										FWHM_arcsec = sampling,
										noise       = self.rms)


		err_data = np.sqrt(e_surfb_d**2 + (0.1*surfb_d)**2)
		xlim = radii_d[np.where(surfb_d < self.rms)][0]*self.px_scale*self.phys_scale*1.2

		x_pix = radii_d*np.cos(PA)
		y_pix = radii_d*np.sin(PA)


		if(INC < 70):

			if(self.type_double == 'exp_bulge'):
				ax2.plot(radii_m1*self.px_scale*self.phys_scale, surfb_m1, linestyle="dashed", color=data_colour[1], label='exp')
				ax2.plot(radii_m2*self.px_scale*self.phys_scale, surfb_m2, linestyle="dotted", color=data_colour[1], label='bulge')
				ax2.vlines(x=self.beam*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='solid', color='k', label='PSF')
				ax2.vlines(x=pmod[r'r$_{eff}$,1']['value']*self.px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='dashed', color='k', label='Reff, 1')
				ax2.vlines(x=pmod[r'r$_{eff}$,2']['value']*self.px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='dotted', color='k', label='Reff, 2')
			elif(self.type_double == 'exp_psf' or self.type_double == 'sersic_psf'):
				ax2.plot(radii_m1*self.px_scale*self.phys_scale, surfb_m1, linestyle="dashed", color=data_colour[1], label='exp')
				ax2.plot(radii_m2*self.px_scale*self.phys_scale, surfb_m2, linestyle="dotted", color=data_colour[1], label='PSF')
				ax2.vlines(x=self.beam*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='solid', color='k', label='PSF')
				ax2.vlines(x=reff*self.px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='dashed', color='k', label='Reff')
			elif(self.type_double == 'exp_sersic'):
				ax2.plot(radii_m1*self.px_scale*self.phys_scale, surfb_m1, linestyle="dashed", color=data_colour[1], label='exp')
				ax2.plot(radii_m2*self.px_scale*self.phys_scale, surfb_m2, linestyle="dotted", color=data_colour[1], label='sersic')
				ax2.vlines(x=self.beam*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='solid', color='k', label='PSF')
				ax2.vlines(x=reff*self.px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='dashed', color='k', label='Reff')

			ax2.errorbar(-99, -99, yerr = err_data[0], fmt="o", color=data_colour[0], label='Data')
			ax2.plot(radii_m*self.px_scale*self.phys_scale, surfb_m, linestyle="solid", color=data_colour[1], label='2D Sersic')
			ax2.hlines(y=self.rms, xmin=0, xmax=xlim, ls=':', color='k', label='RMS')

			for r in range(0, len(radii_d)):
				if(surfb_d[r] > self.rms):
					ax2.errorbar(radii_d[r]*self.px_scale*self.phys_scale, surfb_d[r], yerr = err_data[r], fmt="o", color=data_colour[0])

			ax2.set_ylim(0.9*self.rms, np.amax(surfb_d)*1.5)
			ax2.set_xlim(0, xlim)
			ax2.set_xlabel('Radius (kpc)', fontsize=13)
			ax2.set_ylabel('Surface brightness (mJy/beam km s$^{-1}$)', fontsize=13)

			ax2.yaxis.set_label_position("left")
			ax2.set_yscale("log")
			ax2.legend()


		"""
		else:
			#g2 = gridspec.GridSpec(2,3, left=0.1, bottom=0.1, top=0.94, right=0.99, wspace=0.3) 
			ax21 = plt.subplot(g2[0, 2])
			ax22 = plt.subplot(g2[1, 2])
			ax2.set_ylabel('Surface brightness (mJy/beam km s$^{-1}$)', fontsize=13)

			radii_m, surfb_m, e_surfb_m, rings_m = brightness_edgeon(data      = mod,
									RMAX_arcsec = RMAX,
									XPOS        = Y0,
									YPOS        = X0,
									inc0        = INC,
									pa0         = PA,
									pix_to_arc  = self.px_scale,
									FWHM_arcsec = sampling,
									noise       = self.rms)

			ax21.set_xlim(0, xlim)
			ax21.set_xlabel('Radius (kpc)', fontsize=13)
			ax21.yaxis.set_label_position("left")
			ax21.set_yscale("log")
			ax21.legend()



			ax22.set_xlim(0, xlim)
			ax22.set_xlabel('Z (kpc)', fontsize=13)
			ax22.yaxis.set_label_position("left")
			ax22.set_yscale("log")
			ax22.legend()
		"""

		# cutting maps to zoom in on the galaxy
		S = int(radii_d[np.where(surfb_d < self.rms)][0]*3.0)
		totS = self.data.shape[0]
		rx = X0/totS
		ry = Y0/totS
		a = int((totS - S)*rx)
		b = int((totS - S)*ry)


		wmap = WCS(self.header, naxis=2)
		residual = (self.data-mod)/self.rms

		#wmapcut     = wmap[a:a+S, b:b+S]
		#modelcut    = mod[a:a+S, b:b+S]
		#datacut     = self.data[a:a+S, b:b+S]
		#residualcut = residual[a:a+S, b:b+S]

		wmapcut     = wmap
		modelcut    = mod
		datacut     = self.data
		residualcut = residual


		# Data with model contours
		Cmap1 = matplotlib.cm.get_cmap('gray')
		Cmap1.set_bad(color='black')

		mylevels = np.asarray([2, 4, 8, 16])
		mylevels_neg = np.asarray([-16, -8, -4, -2])


		ax0.axis('off')
		ax0.set_title('Data', fontsize=13)
		AX0 = fig.add_subplot(grid[0], projection=wmapcut)
		im0 = AX0.imshow(datacut, cmap=Cmap1, origin='lower', aspect='equal', interpolation='none')
		AX0.contour(datacut, levels=self.rms*mylevels, linewidths=1, colors=data_colour[0])
		AX0.contour(datacut, levels=self.rms*mylevels_neg, linewidths=0.6, colors='gray', linestyles='dashed')
		AX0.contour(modelcut, levels=self.rms*mylevels, linewidths=1.5, colors=data_colour[1])
		lon = AX0.coords[0]
		lat = AX0.coords[1]
		lon.set_major_formatter('dd:mm:ss.s')
		lat.set_major_formatter('dd:mm:ss.s')
		lon.display_minor_ticks(True)
		lat.display_minor_ticks(True)
		lon.set_axislabel('Right Ascension (J2000)', fontsize=12)
		lat.set_axislabel('Declination (J2000)', fontsize=12)

		AX0.scatter(X0, Y0, s=25, marker='x', color='k')
		AX0.text(0.02, 0.92, self.name, color='white', fontweight='bold', fontsize=13, transform=ax0.transAxes)



		for r in range(0, len(radii_d)):
			if(surfb_d[r] > self.rms):
				AX0.scatter(x_pix[r]  + X0, y_pix[r] + Y0, c=data_colour[0], edgecolors='k', s=25, alpha=0.75)
				AX0.scatter(X0 - x_pix[r] , Y0 - y_pix[r] , c=data_colour[0], edgecolors='k', s=25, alpha=0.75)



		if(self.flags['datatype'] == 'gas'):
			conv = Angle(np.abs(self.header['CDELT1']), u.deg).arcsec
			majb = Angle(np.abs(self.header['BMAJ']), u.deg).arcsec/(2*conv)
			minb = Angle(np.abs(self.header['BMIN']), u.deg).arcsec/(2*conv)
			pab = Angle(np.abs(self.header['BPA']), u.deg).value
		else:
			conv = Angle(np.abs(self.header['CD2_2']), u.deg).arcsec
			majb = Angle(np.abs(self.header['PSFMAJ']), u.arcsec).arcsec/(2*conv)
			minb = Angle(np.abs(self.header['PSFMIN']), u.arcsec).arcsec/(2*conv)
			pab = Angle(np.abs(self.header['PSFPA']), u.deg).value
		posa = np.radians(pab-90)
		t = np.linspace(0,2*np.pi,100)
		dist = 10
		xt = dist+majb*np.cos(posa)*np.cos(t)-minb*np.sin(posa)*np.sin(t)  
		yt = dist+majb*np.sin(posa)*np.cos(t)+minb*np.cos(posa)*np.sin(t)
		AX0.plot(xt, yt, '-', c='k', lw=1.2)
		AX0.fill_between(xt, yt, hatch='/////', fc='white')


		# Residual
		vmin = int(np.nanmin(residualcut))
		vmax = int(np.nanmax(residualcut))

		if(np.abs(vmax)>np.abs(vmin)): NN  = (2*np.abs(vmax))
		else: NN  = (2*np.abs(vmin))
		myN = NN
		vv = NN/2

		white_range_min = (vv-1)/(2*vv)
		white_range_max = (vv+1)/(2*vv)
		mycmap = colors.LinearSegmentedColormap.from_list('mycmap', [(0, '#008080'), (white_range_min, 'white'), (white_range_max, 'white'), (1, '#00008B')], N=myN)
		shifted_cmap = mycmap

		ticks = np.linspace(-vv, vv, 5)


		# Plot residual on the third subplot
		ax1.axis('off')
		ax1.set_title('Residual', fontsize=13)
		ax1 = fig.add_subplot(grid[1], projection=wmapcut)
		im1 = ax1.imshow(residualcut, cmap=shifted_cmap, vmin=-vv, vmax=vv, origin='lower', aspect='equal', interpolation='none')
		lon2 = ax1.coords[0]
		lat2 = ax1.coords[1]
		lon2.set_major_formatter('dd:mm:ss.s')
		lat2.set_major_formatter('dd:mm:ss.s')
		lon2.set_ticklabel_visible(False)
		lat2.set_ticklabel_visible(False)
		lon2.display_minor_ticks(True)
		lat2.display_minor_ticks(True)
		lon2.set_axislabel('')
		lat2.set_axislabel('')

		ax1.scatter(X0, Y0, s=25, marker='x', color='k')


		cbar_ax0 = fig.add_axes([0.35, 0.1265, 0.008, 0.792])
		cbar0 = fig.colorbar(im0, cax=cbar_ax0, orientation='vertical')
		cbar0.set_label('mJy/beam km s$^{-1}$', fontsize=12, labelpad = 0.5)
		cbar_ax0.tick_params(labelsize=9)
		cbar_ax1 = fig.add_axes([0.421, 0.09, 0.248, 0.022])
		cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal', ticks=ticks)
		cbar1.set_label('RMS', fontsize=12, labelpad = 1.0)
		cbar_ax1.tick_params(labelsize=9)


		if(path == None):
			if(self.flags['datatype'] == 'stars'): plt.savefig('./output/dySB/plots/stars/'+self.name+'_sersic_model.png', format='png', dpi=300)
			if(self.flags['datatype'] == 'gas'): plt.savefig('./output/dySB/plots/gas/'+self.name+'_sersic_model.png', format='png', dpi=300)
		else:
			if(self.flags['datatype'] == 'stars'): plt.savefig(path+self.name+'_stars_sersic_model.png', format='png', dpi=300)
			if(self.flags['datatype'] == 'gas'): plt.savefig(path+self.name+'_gas_sersic_model.png', format='png', dpi=300)


class dyDy:
	def __init__(self, model, radii, vel, verr, logmgas, rgas, redshift, name, rtrunc_gas=None, rtrunc_star=None, path=None):

		self.path = path
		self.flags = model['flags'].copy()
		self.pars = model['pars'].copy() #for par in self.keys
		

		keys    = ['mstar','restar','n','alphaco','mdm','cdm']

		self.keys = keys


		if(self.flags['BH'] == 'None'):
			del self.pars['mbh']
		elif(self.flags['BH'] == 'point_mass'):
			self.keys.append('mbh')
		elif(self.flags['BH'] == 'bh_host_local' or self.flags['BH'] == 'bh_host_highz'):
			del self.pars['mbh']


		if(self.flags['star'] == 'bulge_disc'):
			self.keys.append('b/t')
			self.keys.append('redisc')
		elif(self.flags['star'] == 'exp_psf'):
			self.keys.append('b/t')
			del self.pars['n']
			self.keys.remove('n')
		elif(self.flags['star'] == 'sersic_psf'):
			self.keys.append('b/t')
		elif(self.flags['star'] == 'double_exp'):
			self.keys.append('restar2')
			self.keys.append('d1/t')


		if(self.flags['dm'] == 'fbar' or self.flags['dm'] == 'fbar_cmr'):
			self.keys.append('fbar')
			del self.pars['mdm']
			self.keys.remove('mdm')
			self.pars['cdm']['vary'] = False
			#self.pars['fbar'] = {'value': 0.187,  'vary': True,	'prior': {'type': 'fbaryon',  'low': 1e-6,	'up': 0.187}}




		#if(self.flags['dm'] == 'sidm'):
		#	self.keys    = ['mstar','restar','n','alphaco','rho_s','rs', 'rc']
		#	del self.pars['mdm']
		#	del self.pars['cdm']
		#	self.pars['rho_s'] = {'value': 5,  'vary': True,	'prior': {'type': 'uni',  'low': 0.1,	'up': 15}}
		#	self.pars['rs'] = {'value': 3,  'vary': True,	'prior': {'type': 'uni',  'low': 0.1,	'up': 10}}
		#	self.pars['rc'] = {'value': 1,  'vary': True,	'prior': {'type': 'uni',  'low': 0.1,	'up': 10}}


		#self.parname = ['M$_*$', 'R$_{\mathrm{eff,*}}$', '$n_*$', r'M$_{\mathrm{gas}}$', 'M$_{\mathrm{DM}}$', 'c$_{200}$']




		self.radii = radii
		self.vel = vel
		self.verr = verr
		self.rtruncg = rtrunc_gas
		self.rtruncs = rtrunc_star
		self.inv_sigma2 = 1./(verr**2 )
		self.redshift = redshift
		self.name = name

		self.Hz = cosmo.H(redshift).value
		h = self.Hz/100.0
		add = cosmo.angular_diameter_distance(redshift).value
		self.arc2kpc = (add/206265.)*1.e+3

		self.mgas = logmgas
		if self.flags['gas'] == 'exp': self.vgas = V_exp(self.radii, logmgas, rgas, self.arc2kpc)
		elif self.flags['gas'] == 'exp_trunc': self.vgas = V_exp_trunc(self.radii, logmgas, rgas, self.arc2kpc, self.rtruncg)




	def run(self, fit='new'):
	# Project hypercube on prior phase space
	# ----------------------------------------------------------------------
		def ediprior(x):
			
			if self.pars['mstar']['vary']:
				xind = 0
				xppf = []
				for par in self.pars:
					if par=='mstar':
						if self.pars[par]['prior']['type'].lower() in gprior:
							mstar = scipy.stats.norm.ppf(x[xind],loc=self.pars[par]['prior']['loc'],scale=self.pars[par]['prior']['scale'])
						elif self.pars[par]['prior']['type'].lower() in uprior:
							mstar = scipy.stats.uniform.ppf(x[xind],loc=self.pars[par]['prior']['low'],scale=self.pars[par]['prior']['up']-self.pars[par]['prior']['low'])
						elif self.pars[par]['prior']['type'].lower() in ulprior:
							mstar = scipy.stats.uniform.ppf(x[xind],loc=np.log10(self.pars[par]['prior']['low']),scale=np.log10(self.pars[par]['prior']['up'])-np.log10(self.pars[par]['prior']['low']))
						elif self.pars[par]['prior']['type'].lower() in lnprior:
							mstar = scipy.stats.lognorm.ppf(x[xind],loc=self.pars[par]['prior']['loc'], scale = 1.0, s = self.pars[par]['prior']['scale'])
						elif self.pars[par]['prior']['type'].lower() in tprior:
							mstar = scipy.stats.truncnorm.ppf(x[xind],a =self.pars[par]['prior']['a1'], b =self.pars[par]['prior']['a2'], loc=self.pars[par]['prior']['loc'], scale = self.pars[par]['prior']['scale'])

						xind += 1
			else:
				mstar = self.pars['mstar']['value']
			

			xind = 0
			xppf = []
			for par in self.pars:
				if self.pars[par]['vary']:
					if self.pars[par]['prior']['type'].lower() in gprior:
						xppf.append(scipy.stats.norm.ppf(x[xind],loc=self.pars[par]['prior']['loc'],scale=self.pars[par]['prior']['scale']))
					elif self.pars[par]['prior']['type'].lower() in uprior:
						xppf.append(scipy.stats.uniform.ppf(x[xind],loc=self.pars[par]['prior']['low'],scale=self.pars[par]['prior']['up']-self.pars[par]['prior']['low']))
					elif self.pars[par]['prior']['type'].lower() in ulprior:
						xppf.append(10.0**scipy.stats.uniform.ppf(x[xind],loc=np.log10(self.pars[par]['prior']['low']),scale=np.log10(self.pars[par]['prior']['up'])-np.log10(self.pars[par]['prior']['low'])))
					elif self.pars[par]['prior']['type'].lower() in lnprior:
						xppf.append(scipy.stats.lognorm.ppf(x[xind],loc=self.pars[par]['prior']['loc'], scale = 1.0, s = self.pars[par]['prior']['scale']))
					elif self.pars[par]['prior']['type'].lower() in tprior:
						xppf.append(scipy.stats.truncnorm.ppf(x[xind], a=self.pars[par]['prior']['a1'], b =self.pars[par]['prior']['a2'], loc=self.pars[par]['prior']['loc'], scale = self.pars[par]['prior']['scale']))
					elif self.pars[par]['prior']['type'].lower() in ['star']:
						xppf.append(np.log10(scipy.stats.lognorm.ppf(x[xind],loc=mstarmhalo(mstar), scale = 1.0, s = self.pars[par]['prior']['scale'])))
					elif self.pars[par]['prior']['type'].lower() in ['fbaryon']:
#						xppf.append(scipy.stats.uniform.ppf(x[xind],loc=self.pars[par]['prior']['low'],scale=self.pars[par]['prior']['up']-self.pars[par]['prior']['low']))
						xppf.append(scipy.stats.uniform.ppf(x[xind],loc=np.log10(self.pars[par]['prior']['low']),scale=np.log10(self.pars[par]['prior']['up'])-np.log10(self.pars[par]['prior']['low'])))
					xind += 1

			return np.array(xppf)

	# Compute log-likelihood
	# ----------------------------------------------------------------------  
		def logprob(x):
			xind = 0
			pmod = copy.deepcopy(self.pars)
			for par in self.pars: 
				if self.pars[par]['vary']: 
					pmod[par]['value'] = x[xind]
					xind += 1


			if self.flags['star'] == 'sersic':
				vstar = V_sersic(self.radii, pmod['mstar']['value'], pmod['restar']['value'],pmod['n']['value'], self.arc2kpc)
				vstar_2 = vstar**2
			elif self.flags['star'] == 'exp':
				vstar = V_exp(self.radii, pmod['mstar']['value'], pmod['restar']['value']/1.678, self.arc2kpc)
				vstar_2 = vstar**2
			elif self.flags['star'] == 'exp_trunc':
				vstar = V_exp_trunc(self.radii, pmod['mstar']['value'], pmod['restar']['value']/1.678, self.arc2kpc, self.rtruncs)
				vstar_2 = vstar**2
			elif self.flags['star'] == 'bulge_disc':
				stellar_mass = pmod['mstar']['value']
				b_t = pmod['b/t']['value']
				bulge_mass = (10**pmod['mstar']['value'])*pmod['b/t']['value']
				disc_mass = 10**(pmod['mstar']['value']) - bulge_mass

				vbulge = V_sersic(self.radii, np.log10(bulge_mass), pmod['restar']['value'],pmod['n']['value'], self.arc2kpc)

				vdisc = V_exp(self.radii, np.log10(disc_mass), pmod['redisc']['value']/1.678, self.arc2kpc)
				vstar_2 = vbulge**2 + vdisc**2

			elif self.flags['star'] == 'exp_psf':
				stellar_mass = pmod['mstar']['value']
				b_t = pmod['b/t']['value']
				bulge_mass = (10**pmod['mstar']['value'])*b_t
				disc_mass = 10**(pmod['mstar']['value']) - bulge_mass

				vbulge = V_point_source(self.radii, np.log10(bulge_mass), self.arc2kpc)

				vdisc = V_exp(self.radii, np.log10(disc_mass), pmod['restar']['value']/1.678, self.arc2kpc)
				vstar_2 = vbulge**2 + vdisc**2

			elif self.flags['star'] == 'sersic_psf':
				stellar_mass = pmod['mstar']['value']
				b_t = pmod['b/t']['value']
				bulge_mass = (10**pmod['mstar']['value'])*b_t
				disc_mass = 10**(pmod['mstar']['value']) - bulge_mass

				vbulge = V_point_source(self.radii, np.log10(bulge_mass), self.arc2kpc)
				vdisc = V_sersic(self.radii, np.log10(disc_mass), pmod['restar']['value'], pmod['n']['value'], self.arc2kpc)
				vstar_2 = vbulge**2 + vdisc**2

			elif self.flags['star'] == 'double_exp':
				stellar_mass = pmod['mstar']['value']
				d_t = pmod['d1/t']['value']

				disc1_mass = (10**pmod['mstar']['value'])*d_t
				disc2_mass = (10**(pmod['mstar']['value'])) - disc1_mass

				vdisc1 = V_exp(self.radii, np.log10(disc1_mass), pmod['restar']['value']/1.678, self.arc2kpc)
				vdisc2 = V_exp(self.radii, np.log10(disc2_mass), pmod['restar2']['value']/1.678, self.arc2kpc)

				vstar_2 = vdisc1**2 + vdisc2**2



			if self.flags['BH'] == 'point_mass':
				vbh = V_point_source(self.radii, pmod['mbh']['value'], self.arc2kpc)
			elif self.flags['BH'] == 'bh_host_local':
				bhmass = BH_host_local(pmod['mstar']['value'])
				vbh = V_point_source(self.radii, bhmass, self.arc2kpc)
			elif self.flags['BH'] == 'bh_host_highz':
				bhmass = BH_host_highz(pmod['mstar']['value'], self.redshift)
				vbh = V_point_source(self.radii, bhmass, self.arc2kpc)
			elif self.flags['BH'] == 'None':
				vbh = 0


			if self.flags['dm'] == 'nfw':
				vdm = V_nfw(self.radii, pmod['mdm']['value'], pmod['cdm']['value'], self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'nfw_c':
				C200 = mass_c_relation(pmod['mdm']['value'], self.redshift)
				vdm = V_nfw(self.radii, pmod['mdm']['value'], C200, self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'burkert':
				vdm = V_burkert(self.radii, pmod['mdm']['value'], pmod['cdm']['value'], self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'isothermal':
				vdm = V_isothermal(self.radii, pmod['mdm']['value'], pmod['cdm']['value'], self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'nodm':    
				vdm = 0
			elif self.flags['dm'] == 'fbar':
				DMmass = fbaryonmhalo(pmod['mstar']['value'], pmod['alphaco']['value'],self.mgas, pmod['fbar']['value'])
				vdm = V_nfw(self.radii, DMmass, pmod['cdm']['value'], self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'fbar_cmr':
				DMmass = fbaryonmhalo(pmod['mstar']['value'], pmod['alphaco']['value'],self.mgas, pmod['fbar']['value'])
				C200 = mass_c_relation(DMmass, self.redshift)
				vdm = V_nfw(self.radii, DMmass, C200, self.Hz, self.arc2kpc)
			elif self.flags['dm'] == 'sidm':
				vdm = V_sidm(self.radii, pmod['rho_s']['value'], pmod['rs']['value'], pmod['rc']['value'], self.arc2kpc)

			if   self.flags['alpha_co'] == 'const':
				vgas_2 = self.vgas**2*pmod['alphaco']['value']

			
			model = np.sqrt(vstar_2 + vgas_2 + vdm**2 + vbh**2)


			return -0.5*(np.sum((self.vel-model)**2 * self.inv_sigma2))




		if(fit == 'new'):
			print('The fit is set to \'new\', initiating a new fit.')
			self.ndim = np.count_nonzero(np.array([self.pars[par]['vary'] for par in self.keys]))

			with Pool(8, logprob, ediprior) as pool:
				self.sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim=self.ndim, nlive=200, pool=pool)
				self.sampler.run_nested(checkpoint_file = self.name+'_checkpointing.save')

			self.res1 = self.sampler.results

			namechain = self.name+'_dyDy.dill'
			with open(namechain,'wb') as file: dill.dump(self.res1, file) 
			self.wgt1 = np.exp(self.sampler.results['logwt']-self.sampler.results['logz'][-1])

		elif(fit == 'resume'):
			print('The fit is set to \'resume\', resuming the previous fit.')
			if(os.path.exists(self.name+'_checkpointing.save')):
				self.ndim = np.count_nonzero(np.array([self.pars[par]['vary'] for par in self.keys]))
				with Pool(8, logprob, ediprior) as pool:
					self.sampler = dynesty.NestedSampler.restore(self.name+'_checkpointing.save')
					self.sampler.run_nested(resume=True)
				self.res1 = self.sampler.results
				namechain = self.name+'_dyDy.dill'
				with open(namechain,'wb') as file: dill.dump(self.res1, file) 
				self.wgt1 = np.exp(self.sampler.results['logwt']-self.sampler.results['logz'][-1])
			else: print('Checkpointing file does not exist. Please select a different argument for fitting.\nThe options are \'new\' for a new run, \'resume\' for continuing a previous fit or \'read\' for reading an existing finished posterior chain.')

		elif(fit == 'read'):
			print('The fit is set to \'read\', reading the results of a previous fit.')
			namechain = self.name+'_dyDy.dill'
			with open(namechain,'rb') as file: 
				self.res1 = dill.load(file)
			self.ndim = self.res1.samples.shape[1]
			self.wgt1 = np.exp(self.res1['logwt']-self.res1['logz'][-1])








	def plot_dyDy(self):
		self.mean = np.array([dynesty.utils.quantile(self.res1.samples[:,p],[0.50,0.16,0.84],weights=self.wgt1) for p in range(self.res1.samples.shape[1])]).T
		xind = 0

		pmod = copy.deepcopy(self.pars)
		
		for par in self.pars:
			if self.pars[par]['vary']:
				pmod[par]['value'] = self.mean[0][xind]
				xind += 1

		#ROTATION CURVE
		fig1, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
		plot_dydy(ax, self.radii, self.vel, self.verr, self.vgas, pmod, self.flags['BH'], self.flags['star'], self.flags['dm'], self.flags['gas'], self.flags['alpha_co'], self.mgas, self.Hz, self.arc2kpc, self.rtruncs, self.redshift)

		

		if(self.path != None): fig1.savefig(self.path+self.name+'_rc.png',format='png',dpi=300)
		else: fig1.savefig('./output/dyDy/'+self.name+'_rc.png',format='png',dpi=300)


		#CORNER PLOT

		#self.res1.samples[:,3] = self.res1.samples[:,3]*(10**self.mgas)/(1e10)

		fig2, axes = plt.subplots(self.ndim, self.ndim, figsize=(10,10))

		xind = 0
		self.labels = []
		self.bestfit = {}
		for par in self.pars:
			if self.pars[par]['vary']:
				if(xind < self.ndim):
					self.labels.append(par)
					self.bestfit.update({par: self.res1.samples[:,xind]})
					xind += 1	

		



		fig, ax = dynesty.plotting.cornerplot(self.res1, labels=self.labels, label_kwargs={'fontsize' : 13}, title_quantiles=(0.16, 0.5, 0.84), quantiles=(0.16, 0.5, 0.84), truth_kwargs={'fontsize' : 13}, fig=(fig2, axes[:,:self.ndim]), show_titles=True, hist_kwargs={'alpha': 0.4})

		plt.suptitle(self.name+'_dyDy')

		if(self.path != None): fig2.savefig(self.path+self.name+'_dycorner.png',format='png',dpi=300)
		else: fig2.savefig('./output/dyDy/'+self.name+'_dycorner.png',format='png',dpi=300)

		plt.close()



class outputs:
	def __init__(self, full_data):
		self.galID       = full_data.ID
		self.phys_scale  = full_data.phys_scale
		self.data_gas    = full_data.mom0gas
		self.header_gas  = full_data.ALMA_header
		self.rms_gas     = full_data.rms_gas
		self.wcs_gas     = WCS(full_data.ALMA_header, naxis=2)
		self.beam_gas    = self.header_gas['BMIN']
		self.gas_px_scale = full_data.gas_px_scale

		if(full_data.JWST_flag == 1):
			self.data_stars     = full_data.JWST_data
			self.rms_stars      = full_data.rms_JWST
			self.wcs_stars      = WCS(full_data.JWST_header, naxis=2)
			self.header_stars   = full_data.JWST_header
			self.stars_px_scale = full_data.JWST_px_scale
			self.beam_stars     = self.header_stars['PSFMIN']
		else:
			self.data_stars     = full_data.HST_data
			self.rms_stars      = full_data.rms_HST
			self.wcs_stars      = WCS(full_data.HST_header, naxis=2)
			self.header_stars   = full_data.HST_header
			self.stars_px_scale = full_data.HST_px_scale
			self.beam_stars     = self.header_stars['PSFMIN']

		self.model_gas   = fits.open('./output/dySB/models/gas/ID'+str(self.galID)+'_gas_sersic_model.fits')[0].data
		self.model_stars = fits.open('./output/dySB/models/stars/ID'+str(self.galID)+'_stars_sersic_model.fits')[0].data


		# Opening chain results
		gas_file = './output/dySB/chains/dySB_chains_with_weights_ID'+str(self.galID)+'_gas.txt'
		stars_file = './output/dySB/chains/dySB_chains_with_weights_ID'+str(self.galID)+'_stars.txt'

		df_gas = read_file_into_pd(gas_file, data_type='gas')
		df_stars = read_file_into_pd(stars_file, data_type='stars')

		# reading best-fit pars

		self.reff_gas = weighted_quantile(df_gas['r$_{eff}$'], df_gas['weights'], q=0.5)
		self.x0_gas   = weighted_quantile(df_gas['x0'], df_gas['weights'], q=0.5)
		self.y0_gas   = weighted_quantile(df_gas['y0'], df_gas['weights'], q=0.5) 
		self.eps_gas  = weighted_quantile(df_gas['ϵ'], df_gas['weights'], q=0.5) 
		self.pa_gas   = weighted_quantile(df_gas['PA'], df_gas['weights'], q=0.5) 

		self.reff_stars = weighted_quantile(df_stars['r$_{eff}$'], df_stars['weights'], q=0.5)
		self.x0_stars   = weighted_quantile(df_stars['x0'], df_stars['weights'], q=0.5)
		self.y0_stars   = weighted_quantile(df_stars['y0'], df_stars['weights'], q=0.5) 
		self.eps_stars  = weighted_quantile(df_stars['ϵ'], df_stars['weights'], q=0.5) 
		self.pa_stars   = weighted_quantile(df_stars['PA'], df_stars['weights'], q=0.5) 


		# modifying the chains:
		df_gas['I$_{e,gas}$'] = df_gas['I$_{e,gas}$'] * 1000    # Jy to mJy
		df_gas['r$_{eff}$'] = df_gas['r$_{eff}$'] * self.gas_px_scale * self.phys_scale    # px to kpc
		df_gas['x0'] = self.wcs_gas.wcs_pix2world(df_gas['x0'], df_gas['y0'], 0)[0]  # px to RA
		df_gas['y0'] = self.wcs_gas.wcs_pix2world(df_gas['x0'], df_gas['y0'], 0)[1]  # px to DEC
		df_gas['PA'] = np.rad2deg(df_gas['PA'])                  # rad to deg

		df_stars['r$_{eff}$'] = df_stars['r$_{eff}$'] * self.stars_px_scale * self.phys_scale     # px to kpc
		df_stars['x0'] = self.wcs_stars.wcs_pix2world(df_stars['x0'], df_stars['y0'], 0)[0]  # px to RA
		df_stars['y0'] = self.wcs_stars.wcs_pix2world(df_stars['x0'], df_stars['y0'], 0)[1]  # px to DEC
		df_stars['PA'] = np.rad2deg(df_stars['PA'])  # rad to deg

		self.df_gas   = df_gas
		self.df_stars = df_stars





	def plot_dySB_profile(self):
		print('# Plotting dySB Main Figures: 2D and 1D profiles')

		# Create a figure and gridspec layout
		fig = plt.figure(figsize=(7.5, 4.5))


		gs = GridSpec(1, 2, width_ratios=[2.2,1], wspace=0.2, top=0.95, bottom=0.07, right=0.98, left=0.085)

		gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], hspace=0.2, wspace=0.25)
		gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.2)


		# Define axes for the panels
		ax1  = fig.add_subplot(gs1[0, 0])
		AX1  = fig.add_subplot(gs1[0, 0], projection=self.wcs_gas)
		lon1 = AX1.coords[0]
		lat1 = AX1.coords[1]
		ax2  = fig.add_subplot(gs1[0, 1])
		AX2  = fig.add_subplot(gs1[0, 1], projection=self.wcs_gas)
		lon2 = AX2.coords[0]
		lat2 = AX2.coords[1]
		ax3  = fig.add_subplot(gs2[0, 0])
		ax4  = fig.add_subplot(gs1[1, 0])
		AX4  = fig.add_subplot(gs1[1, 0], projection=self.wcs_stars)
		lon4 = AX4.coords[0]
		lat4 = AX4.coords[1]
		ax5  = fig.add_subplot(gs1[1, 1])
		AX5  = fig.add_subplot(gs1[1, 1], projection=self.wcs_stars)
		lon5 = AX5.coords[0]
		lat5 = AX5.coords[1]
		ax6  = fig.add_subplot(gs2[1, 0])

		# Defining global properties of the plots
		levels = np.asarray([2, 4, 8, 16])
		levels_neg = np.asarray([-16, -8, -4, -2])
		Cmap1 = matplotlib.cm.get_cmap('gray')
		Cmap1.set_bad(color='black')

		#contours: gas data, gas data neg, gas model, star data, star data neg, star model, misc
		Colours = ['#1be7ff', '#b3fffc', '#fb62f6', '#26f7bf', '#64fcd4', '#fb5607', 'k']





		######################################################################
		# Plot first row: gas fit

		# Data with contours
		ax1.axis('off')
		ax1.set_title('Data and Model', fontsize=10)
		im1 = AX1.imshow(self.data_gas*1000, cmap=Cmap1, origin='lower', aspect='equal', interpolation='none')
		AX1.contour(self.data_gas, levels=self.rms_gas*levels, linewidths=1, colors=Colours[0])
		AX1.contour(self.data_gas, levels=self.rms_gas*levels_neg, linewidths=0.6, colors=Colours[1], linestyles='dashed')
		AX1.contour(self.model_gas, levels=self.rms_gas*levels, linewidths=1.5, colors=Colours[2])
		lon1.set_major_formatter('dd:mm:ss.s')
		lat1.set_major_formatter('dd:mm:ss.s')
		lon1.set_ticklabel(size=6)
		lat1.set_ticklabel(size=6)
		lon1.set_ticks(spacing=2* u.arcsec)
		lat1.set_ticks(spacing=2* u.arcsec)
		lon1.display_minor_ticks(True)
		lat1.display_minor_ticks(True)
		lon1.set_axislabel('Right Ascension (J2000)', fontsize=7, minpad=0.7)
		lat1.set_axislabel('Declination (J2000)', fontsize=7, minpad=0.5)

		AX1.text(0.05, 0.98, 'Gas', color='white', fontsize=9, horizontalalignment='left', verticalalignment='top', fontweight='bold', transform=ax1.transAxes)

		conv = Angle(np.abs(self.header_gas['CDELT1']), u.deg).arcsec
		majb = Angle(np.abs(self.header_gas['BMAJ']), u.deg).arcsec/(2*conv)
		minb = Angle(np.abs(self.header_gas['BMIN']), u.deg).arcsec/(2*conv)
		pab = Angle(self.header_gas['BPA'], u.deg).value
		posa = np.radians(pab-90)
		t = np.linspace(0,2*np.pi,100)
		dist = 20

		xt = dist+majb*np.cos(posa)*np.cos(t)-minb*np.sin(posa)*np.sin(t)  
		yt = dist+majb*np.sin(posa)*np.cos(t)+minb*np.cos(posa)*np.sin(t)

		AX1.plot(xt, yt, '-', c='white', lw=1.2)
		#AX1.fill_between(xt, yt, hatch='///', fc='white')


		AX1.scatter(self.x0_gas, self.y0_gas, s=25, marker='x', color=Colours[6])



		# Normalised residual map
		residual = (self.data_gas-self.model_gas)/self.rms_gas
		vmin = int(np.nanmin(residual))
		vmax = int(np.nanmax(residual))
		if(np.abs(vmax)>np.abs(vmin)): NN  = (2*np.abs(vmax))
		else: NN  = (2*np.abs(vmin))
		myN = NN
		vv = NN/2

		white_range_min = (vv-1)/(2*vv)
		white_range_max = (vv+1)/(2*vv)
		mycmap = colors.LinearSegmentedColormap.from_list('mycmap', [(0, '#008080'), (white_range_min, 'white'), (white_range_max, 'white'), (1, '#00008B')], N=myN)
		shifted_cmap = mycmap

		ticks_g = np.linspace(-vv, vv, 5)



		# Plot residual on the third subplot
		ax2.axis('off')
		ax2.set_title('Residual', fontsize=10)
		im2 = AX2.imshow(residual, cmap=shifted_cmap, vmin=-vv, vmax=vv, origin='lower', aspect='equal', interpolation='none')
		lon2.set_major_formatter('dd:mm:ss.s')
		lat2.set_major_formatter('dd:mm:ss.s')
		lon2.set_ticklabel_visible(False)
		lat2.set_ticklabel_visible(False)
		lon2.set_ticklabel(size=7)
		lat2.set_ticklabel(size=7)
		lon2.display_minor_ticks(True)
		lat2.display_minor_ticks(True)
		lon2.set_axislabel('')
		lat2.set_axislabel('')



		# Plot 1D profile of data and model
		ax3.set_title('Average 1D Profile', fontsize=10)


		##########################################
		# Plot second row: stellar fit
		
		# Data with contours
		ax4.axis('off')
		im4 = AX4.imshow(self.data_stars, cmap=Cmap1, origin='lower', aspect='equal', interpolation='none')
		AX4.contour(self.data_stars, levels=self.rms_stars*levels, linewidths=1, colors=Colours[3])
		AX4.contour(self.data_stars, levels=self.rms_stars*levels_neg, linewidths=0.6, colors=Colours[4], linestyles='dashed')
		AX4.contour(self.model_stars, levels=self.rms_stars*levels, linewidths=1.5, colors=Colours[5])
		lon4.set_major_formatter('dd:mm:ss.s')
		lat4.set_major_formatter('dd:mm:ss.s')
		lon4.display_minor_ticks(True)
		lat4.display_minor_ticks(True)
		lon4.set_ticklabel(size=6)
		lat4.set_ticklabel(size=6)
		lon4.set_ticks(spacing=2* u.arcsec)
		lat4.set_ticks(spacing=2* u.arcsec)
		lon4.set_axislabel('Right Ascension (J2000)', fontsize=7, minpad=0.7)
		lat4.set_axislabel('Declination (J2000)', fontsize=7, minpad=0.5)

		AX4.text(0.05, 0.98, 'Stars', color='white', fontsize=9, horizontalalignment='left', verticalalignment='top', fontweight='bold', transform=ax4.transAxes)


		conv = Angle(np.abs(self.header_stars['CD2_2']), u.deg).arcsec
		majb = Angle(np.abs(self.header_stars['PSFMAJ']), u.arcsec).arcsec/(2*conv)
		minb = Angle(np.abs(self.header_stars['PSFMIN']), u.arcsec).arcsec/(2*conv)
		pab = Angle(0, u.deg).value
		posa = np.radians(pab-90)
		t = np.linspace(0,2*np.pi,100)
		dist = 10

		xt = dist+majb*np.cos(posa)*np.cos(t)-minb*np.sin(posa)*np.sin(t)  
		yt = dist+majb*np.sin(posa)*np.cos(t)+minb*np.cos(posa)*np.sin(t)

		AX4.plot(xt, yt, '-', c='white', lw=1.2)
		#AX4.fill_between(xt, yt, hatch='///', fc='white')

		AX4.scatter(self.x0_stars, self.y0_stars, s=25, marker='x', color=Colours[6])



		# Normalised residual map
		residual2 = (self.data_stars-self.model_stars)/self.rms_stars
		vmin = int(np.nanmin(residual2))
		vmax = int(np.nanmax(residual2))
		if(np.abs(vmax)>np.abs(vmin)): NN  = (2*np.abs(vmax))
		else: NN  = (2*np.abs(vmin))
		myN = NN
		vv = NN/2

		white_range_min = (vv-1)/(2*vv)
		white_range_max = (vv+1)/(2*vv)
		mycmap = colors.LinearSegmentedColormap.from_list('mycmap', [(0, '#008080'), (white_range_min, 'white'), (white_range_max, 'white'), (1, '#00008B')], N=myN)
		shifted_cmap = mycmap

		ticks_s = np.linspace(-vv, vv, 5)


		# Plot residual on the third subplot
		ax5.axis('off')
		im5 = AX5.imshow(residual2, cmap=shifted_cmap, vmin=-vv, vmax=vv, origin='lower', aspect='equal', interpolation='none')
		lon5.set_major_formatter('dd:mm:ss.s')
		lat5.set_major_formatter('dd:mm:ss.s')
		lon5.set_ticklabel_visible(False)
		lat5.set_ticklabel_visible(False)
		lon5.set_ticklabel(size=5)
		lat5.set_ticklabel(size=5)
		lon5.display_minor_ticks(True)
		lat5.display_minor_ticks(True)
		lon5.set_axislabel('')
		lat5.set_axislabel('')


		# Colourbars for panels 1 and 4
		cbar_ax1 = fig.add_axes([0.337, 0.549, 0.008, 0.4])
		cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='vertical')
		cbar1.set_label('mJy/beam km s$^{-1}$', fontsize=7, labelpad = 0)
		cbar_ax1.tick_params(labelsize=6, pad=0.2)

		cbar_ax4 = fig.add_axes([0.337, 0.069, 0.008, 0.4])
		cbar4 = fig.colorbar(im4, cax=cbar_ax4, orientation='vertical')
		cbar4.set_label('', fontsize=7, labelpad = 0)
		cbar_ax4.tick_params(labelsize=6, pad=0.2)

		# Colourbars for panels 2 and 5
		cbar_ax2 = fig.add_axes([0.4, 0.53, 0.24, 0.01])
		cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal', ticks=ticks_g)
		cbar2.set_label('RMS', fontsize=7, labelpad = 0.5)
		cbar_ax2.tick_params(labelsize=6, pad=0.3)

		cbar_ax5 = fig.add_axes([0.40, 0.05, 0.24, 0.01])
		cbar5 = fig.colorbar(im5, cax=cbar_ax5, orientation='horizontal', ticks=ticks_s)
		cbar5.set_label('RMS', fontsize=7, labelpad = 0.5)
		cbar_ax5.tick_params(labelsize=6, pad=0.3)

		# 1D profile
		sampling_gas = (self.beam_gas/1.)*3600

		rmax_gas = self.reff_gas*self.gas_px_scale*5.0
		inc_gas  = inc(self.eps_gas)

		radii_d, surfb_d, e_surfb_d, rings_d = brightness(data      = self.data_gas*1000,
								RMAX_arcsec = rmax_gas,
								XPOS        = self.y0_gas,
								YPOS        = self.x0_gas,
								inc0        = inc_gas,
								pa0         = self.pa_gas,
								pix_to_arc  = self.gas_px_scale,
								FWHM_arcsec = sampling_gas,
								noise       = self.rms_gas*1000)

		radii_m, surfb_m, e_surfb_m, rings_m = brightness(data      = self.model_gas*1000,
								RMAX_arcsec = rmax_gas,
								XPOS        = self.y0_gas,
								YPOS        = self.x0_gas,
								inc0        = inc_gas,
								pa0         = self.pa_gas,
								pix_to_arc  = self.gas_px_scale,
								FWHM_arcsec = sampling_gas,
								noise       = self.rms_gas*1000)


		rms = self.rms_gas*1000

		err_data = np.sqrt(e_surfb_d**2 + (0.1*surfb_d)**2)

		ax3.errorbar(-99, -99, yerr = err_data[0], fmt="o", color=Colours[0], label='Data')
		ax3.plot(radii_m*self.gas_px_scale*self.phys_scale, surfb_m, linestyle="solid", color=Colours[2], label='2D Sersic', zorder=10)

		xlim = radii_d[np.where(surfb_d < rms)][0]*self.gas_px_scale*self.phys_scale*1.2

		ax3.hlines(y=rms, xmin=0, xmax=xlim, ls=':', lw=1,color='k')

		ax3.vlines(x=self.reff_gas*self.gas_px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='-.', color='k',lw=1)

		for r in range(0, len(radii_d)):
			if(surfb_d[r] > rms):
				ax3.errorbar(radii_d[r]*self.gas_px_scale*self.phys_scale, surfb_d[r], yerr = err_data[r], fmt="o", color=Colours[0])

		ax3.set_ylim(0.9*rms, np.amax(surfb_d)*1.5)
		ax3.set_xlim(0, xlim)
		ax3.set_xlabel('Radius (kpc)', fontsize=8, labelpad=1.2)
		ax3.set_ylabel('Surf. brightness (mJy/beam km s$^{-1}$)', fontsize=8, labelpad=0.5)
		ax3.tick_params(axis='both', which='both', labelsize=6.5, pad=0.5)

		ax3.ticklabel_format(axis='y', style= 'plain')


		ax3.yaxis.set_label_position("left")
		ax3.set_yscale("log")
		ax3.legend(fontsize=7)

		x_pix = radii_d*np.cos(self.pa_gas)
		y_pix = radii_d*np.sin(self.pa_gas)

		for r in range(0, len(radii_d)):
			if(surfb_d[r] > rms):
				AX1.scatter(x_pix[r]  + self.x0_gas, y_pix[r] + self.y0_gas, c=Colours[0], edgecolors='k', s=20, alpha=0.75, zorder=10)
				AX1.scatter(self.x0_gas - x_pix[r] , self.y0_gas - y_pix[r] , c=Colours[0], edgecolors='k', s=20, alpha=0.75, zorder=10)


		###################################
		# for stars

		sampling_stars = (self.beam_stars*2.0)
		rmax_stars = self.reff_stars*self.stars_px_scale*5.0
		inc_stars  = inc(self.eps_gas)

		radii_d, surfb_d, e_surfb_d, rings_d = brightness(data      = self.data_stars,
								RMAX_arcsec = rmax_stars,
								XPOS        = self.y0_stars,
								YPOS        = self.x0_stars,
								inc0        = inc_stars,
								pa0         = self.pa_stars,
								pix_to_arc  = self.stars_px_scale,
								FWHM_arcsec = sampling_stars,
								noise       = self.rms_stars)

		radii_m, surfb_m, e_surfb_m, rings_m = brightness(data      = self.model_stars,
								RMAX_arcsec = rmax_stars,
								XPOS        = self.y0_stars,
								YPOS        = self.x0_stars,
								inc0        = inc_stars,
								pa0         = self.pa_stars,
								pix_to_arc  = self.stars_px_scale,
								FWHM_arcsec = sampling_stars,
								noise       = self.rms_stars)


		err_data = np.sqrt(e_surfb_d**2 + (0.1*surfb_d)**2)

		ax6.errorbar(-99, -99, yerr = err_data[0], fmt="o", color=Colours[3], label='Data')
		ax6.plot(radii_m*self.stars_px_scale*self.phys_scale, surfb_m, linestyle="solid", color=Colours[5], label='2D Sersic', zorder=10)

		xlim = radii_d[np.where(surfb_d < self.rms_stars)][0]*self.stars_px_scale*self.phys_scale*1.2

		ax6.hlines(y=self.rms_stars, xmin=0, xmax=xlim, ls=':', lw=1, color='k', label='RMS')

		ax6.vlines(x=self.reff_stars*self.stars_px_scale*self.phys_scale, ymin=0, ymax=np.amax(surfb_d)*3.0, ls='-.',lw=1, color='k', label='Reff')

		for r in range(0, len(radii_d)):
			if(surfb_d[r] > self.rms_stars):
				ax6.errorbar(radii_d[r]*self.stars_px_scale*self.phys_scale, surfb_d[r], yerr = err_data[r], fmt="o", color=Colours[3])

		ax6.set_ylim(0.9*self.rms_stars, np.amax(surfb_d)*1.5)
		ax6.set_xlim(0, xlim)
		ax6.set_xlabel('Radius (kpc)', fontsize=8, labelpad=1.2)
		ax6.set_ylabel('Surf. brightness ', fontsize=8, labelpad=0.5)
		ax6.tick_params(labelsize=6.5, pad=0.5)


		ax6.yaxis.set_label_position("left")
		ax6.set_yscale("log")
		ax6.legend(fontsize=7)


		x_pix = radii_d*np.cos(self.pa_stars)
		y_pix = radii_d*np.sin(self.pa_stars)

		for r in range(0, len(radii_d)):
			if(surfb_d[r] > self.rms_stars):
				AX4.scatter(x_pix[r]  + self.x0_stars, y_pix[r] + self.y0_stars, c=Colours[3], edgecolors='k', s=20, alpha=0.75, zorder=10)
				AX4.scatter(self.x0_stars - x_pix[r] , self.y0_stars - y_pix[r] , c=Colours[3], edgecolors='k', s=20, alpha=0.75, zorder=10)


		if(self.galID == 1): plt.savefig('./output/figures/Figure3.png', dpi=400)
		else: plt.savefig('./output/figures/appendix/dySB/dySB_ID'+str(self.galID)+'.pdf')



	def plot_dySB_corner(self, path=None):
		print('# Plotting dySB: cornerplots')

		parnames_gas = self.df_gas.columns[:-1].tolist()
		chains_gas  = self.df_gas.drop(columns=['weights']).values
		weights_gas = self.df_gas['weights'].values
		samples_gas = MCSamples(samples=chains_gas, weights=weights_gas, names=parnames_gas, label='Gas')


		g = plots.get_subplot_plotter(width_inch=5)
		g.settings.colormap='cool'
		g.settings.axes_labelsize = 12
		g.settings.axes_fontsize = 4
		g.settings.title_limit_fontsize = 8
		g.settings.title_limit_labels = False


		g.triangle_plot([samples_gas], shaded=True, line_args=[{'ls':'solid', 'color':'black'}], contour_colors=['white'], title_limit=1)


		axes = g.subplots
		i = 0
		for par in parnames_gas:
			p2 = weighted_quantile(self.df_gas[par], self.df_gas['weights'], q=0.02)
			p50 = weighted_quantile(self.df_gas[par], self.df_gas['weights'], q=0.5)
			p84 = weighted_quantile(self.df_gas[par], self.df_gas['weights'], q=0.84)
			p98 = weighted_quantile(self.df_gas[par], self.df_gas['weights'], q=0.98)

			if(par == 'x0' or par == 'y0'):
				l50 = degrees_to_hexagesimal(p50)
				axes[5,i].set_xticks(ticks=[p50])
				axes[5,i].set_xticklabels([l50])
				err = np.round((p84-p50)*3600, 3)
				axes[i,i].set_title(l50+' $\pm$ '+str(err)+'\'', fontsize=4)
				if(i>0):
					axes[i,0].set_yticks(ticks=[p50])
					axes[i,0].set_yticklabels([l50])
					axes[i,0].set_ylabel('')

			else:
				l2  = str(np.round(p2,2))
				l98 = str(np.round(p98,2))
				axes[5,i].set_xticks(ticks=[p2, p98])
				axes[5,i].set_xticklabels([l2, l98])

				if(i>0):
					axes[i,0].set_yticks(ticks=[p2, p98])
					axes[i,0].set_yticklabels([l2, l98])
					axes[i,0].set_ylabel('')
			i = i+1


		if(path == None): plt.savefig('./output/figures/appendix/dySB/dySB_ID'+str(self.galID)+'_corner_gas.pdf', dpi=300)
		else: plt.savefig(path+'dySB_ID'+str(self.galID)+'_corner_gas.pdf', dpi=300)





		parnames_stars = self.df_stars.columns[:-1].tolist()
		chains_stars   = self.df_stars.drop(columns=['weights']).values
		weights_stars  = self.df_stars['weights'].values
		samples_stars  = MCSamples(samples=chains_stars, weights=weights_stars, names=parnames_stars, label='Stars')

		g = plots.get_subplot_plotter(width_inch=5)
		g.settings.colormap='spring'
		g.settings.axes_labelsize = 12
		g.settings.axes_fontsize = 4
		g.settings.title_limit_fontsize = 8
		g.settings.title_limit_labels = False
		g.settings.tight_layout = True
		g.triangle_plot([samples_stars], shaded=True, line_args=[{'ls':'solid', 'color':'black'}], contour_colors=['white'], title_limit=1)



		axes = g.subplots
		i = 0
		for par in parnames_stars:
			p2 = weighted_quantile(self.df_stars[par], self.df_stars['weights'], q=0.02)
			p50 = weighted_quantile(self.df_stars[par], self.df_stars['weights'], q=0.5)
			p84 = weighted_quantile(self.df_stars[par], self.df_stars['weights'], q=0.84)
			p98 = weighted_quantile(self.df_stars[par], self.df_stars['weights'], q=0.98)

			if(par == 'x0' or par == 'y0'):
				l50 = degrees_to_hexagesimal(p50)
				axes[5,i].set_xticks(ticks=[p50])
				axes[5,i].set_xticklabels([l50])
				err = np.round((p84-p50)*3600, 3)
				axes[i,i].set_title(l50+' $\pm$ '+str(err)+'\'', fontsize=4)
				if(i>0):
					axes[i,0].set_yticks(ticks=[p50])
					axes[i,0].set_yticklabels([l50])
					axes[i,0].set_ylabel('')

			else:
				l2  = str(np.round(p2,2))
				l98 = str(np.round(p98,2))
				axes[5,i].set_xticks(ticks=[p2, p98])
				axes[5,i].set_xticklabels([l2, l98])

				if(i>0):
					axes[i,0].set_yticks(ticks=[p2, p98])
					axes[i,0].set_yticklabels([l2, l98])
					axes[i,0].set_ylabel('')
			i = i+1


		if(path == None): plt.savefig('./output/figures/appendix/dySB/dySB_ID'+str(self.galID)+'_corner_stars.pdf', dpi=300)
		else: plt.savefig(path+'dySB_ID'+str(self.galID)+'_corner_stars.pdf', dpi=300)







	def table_dySB(self):
		print('# Creating table with dySB results: {data_type}')

		self.df_gas.columns
		self.df_stars.columns

		if(self.galID == 1):
			with open("Table1_dySB_gas.dat", "w") as myfile:
			    myfile.write('ID & $I_{\mathrm{eff}}$ & $R_{\mathrm{eff}}$  & $\epsilon$ & PA \\\\ \n & (Jy/beam km s$^{-1}$) & (kpc) & & (deg) \\\\ \hline \\noalign{\\vskip 1mm}')
			with open("Table_dySB_stars.dat", "w") as myfile:
			    myfile.write('ID & $I_{\mathrm{eff}}$ & $R_{\mathrm{eff}}$  & $\epsilon$ & PA & Filter \\\\ \n & (units?) & (kpc) & & (deg) & \\\\ \hline \\noalign{\\vskip 1mm}')


		#print('################################################')
		#print('############## DYSB GAS EMISSION ###############')
		#print('################################################')

		gal  = self.galID
		I16  = self.df_gas['I$_{e,gas}$'].quantile(0.16)
		I50  = np.round(self.df_gas['I$_{e,gas}$'].quantile(0.5), 4)
		I84  = self.df_gas['I$_{e,gas}$'].quantile(0.84)
		I_u  = np.round(I84-I50, 4)
		I_d  = np.round(I50-I16, 4)

		R16  = self.df_gas['r$_{eff}$'].quantile(0.16)
		R50  = np.round(self.df_gas['r$_{eff}$'].quantile(0.5),2)
		R84  = self.df_gas['r$_{eff}$'].quantile(0.84)
		R_u  = np.round(R84-R50, 2)
		R_d  = np.round(R50-R16, 2)

		e16  = self.df_gas['ϵ'].quantile(0.16)
		e50  = np.round(self.df_gas['ϵ'].quantile(0.5),2)
		e84  = self.df_gas['ϵ'].quantile(0.84)
		e_u  = np.round(e84-e50, 2)
		e_d  = np.round(e50-e16, 2)

		pa16 = self.df_gas['PA'].quantile(0.16)
		pa50 = np.round(self.df_gas['PA'].quantile(0.5), 2)
		pa84 = self.df_gas['PA'].quantile(0.84)
		pa_u = np.round(pa84-pa50, 2)
		pa_d = np.round(pa50-pa16, 2)


		out_string = str(gal) + ' & '+str(I50)+' $^{+'+str(I_d)+'} _{-'+str(I_u)+'}$ & '+str(R50)+' $^{+'+str(R_d)+'} _{-'+str(R_u)+'}$ & '+str(e50)+' $^{+'+str(e_d)+'} _{-'+str(e_u)+'}$ & '+str(pa50)+' $^{+'+str(pa_d)+'} _{-'+str(pa_u)+'}$ \\\\ \\noalign{\\vskip 1mm} '


		with open("Table1_dySB_gas.dat",'a') as f:
			f.write(out_string+' \n')


		###############################################################################
		#print('################################################')
		#print('############ DYSB STELLAR CONTINUUM ############')
		#print('################################################')

		gal  = self.galID
		I16  = self.df_stars['I$_{e,gas}$'].quantile(0.16)
		I50  = self.df_stars['I$_{e,gas}$'].quantile(0.5)
		I84  = self.df_stars['I$_{e,gas}$'].quantile(0.84)
		R16  = self.df_stars['r$_{eff}$'].quantile(0.16)
		R50  = self.df_stars['r$_{eff}$'].quantile(0.5)
		R84  = self.df_stars['r$_{eff}$'].quantile(0.84)
		e16  = self.df_stars['ϵ'].quantile(0.16)
		e50  = self.df_stars['ϵ'].quantile(0.5)
		e84  = self.df_stars['ϵ'].quantile(0.84)
		pa16 = self.df_stars['PA'].quantile(0.16)
		pa50 = self.df_stars['PA'].quantile(0.5)
		pa84 = self.df_stars['PA'].quantile(0.84)
		#print(int(gal), ' & ', I50 , '$^{+', np.round(I84-I50, 2), '} _{-', np.round(I50-I16,2) ,'}$ & ', np.round(R50,2) , '$^{+', np.round(R84-R50, 2), '} _{-', np.round(R50-R16,2) ,'}$ & ', e50 , '$^{+', np.round(e84-e50, 2), '} _{-', np.round(e50-e16,2) ,'}$ & ', pa50 , '$^{+', np.round(pa84-pa50, 2), '} _{-', np.round(pa50-pa16,2) ,'}$ & ', fw, ' \\\\ \\noalign{\\vskip 1mm} ')







######################################################################
######################################################################
######################################################################
# MISCELANEOUS FUNCTIONS:

def degrees_to_hexagesimal(deg_float):
    degrees = int(deg_float)  # Get the integer part for degrees
    minutes_float = (deg_float - degrees) * 60  # Convert the fractional part into minutes
    minutes = int(minutes_float)  # Get the integer part for minutes
    seconds = (minutes_float - minutes) * 60  # Convert the remaining fractional part into seconds

    # Format the string to include 'deg', 'armin', and 'arcsec'
    hexagesimal_str = f"{degrees}\N{DEGREE SIGN}{minutes}\"{seconds:.1f}\'"
    return hexagesimal_str

def weighted_quantile(data, weights, q):
    """
    q : quantile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(q, cdf, data)


def read_file_into_pd(file_name, data_type):
	with open(file_name, 'r') as file:
	    header_line = file.readline().strip()
	column_names = [name.strip() for name in header_line.split()[1:]]
	df = pd.read_csv(file_name, sep='\s+', comment='#', header=None, skipinitialspace=True)
	column_names[0] = 'I$_{e,'+data_type+'}$'
	df.columns = column_names
	return df


def source_finder(image, threshold_value, name):
	binary_image = image > threshold_value

	labeled_image, num_features = measure.label(binary_image, background=0, return_num=True)

	sources = []
	for region in measure.regionprops(labeled_image, intensity_image=image):
		if region.area < 5:
			continue
        
		diameter = np.sqrt(4 * region.area / np.pi)
		if diameter < 2.5:
			continue

		y_centroid, x_centroid = region.centroid
		sources.append(np.array([x_centroid, y_centroid]))

	brightest_diameter = 0
	for region in measure.regionprops(labeled_image, intensity_image=image):
		if region.max_intensity == region.mean_intensity:
			diameter = np.sqrt(4 * region.area / np.pi)
			if diameter > brightest_diameter:
				brightest_diameter = diameter

	min_separation = 1*brightest_diameter

	image_center = np.array(image.shape) / 2
	sources = sorted(sources, key=lambda source: np.linalg.norm(source - image_center))

	filtered_sources = []
	for source in sources:
		if all(np.linalg.norm(source - other_source) > min_separation for other_source in filtered_sources):
			filtered_sources.append(source)

	print(np.array(filtered_sources))

	# Plotting
	Fig, ax = plt.subplots()
	ax.imshow(image, origin='lower', norm=PowerNorm(0.2))  # Display the image
	source_radius = image.shape[0]*0.1
	for j in range(0, np.array(filtered_sources).shape[0]):
		if(j == 0):
			lw = 2.5
			color = 'red'
		else:
			lw = 1.5
			color = 'yellow'
		circle = Circle(np.array(filtered_sources)[j], source_radius, edgecolor=color, linewidth=lw, facecolor='none')
		ax.add_patch(circle)

	plt.savefig('./output/figures/others/'+name+'_sources.png')


	return np.array(filtered_sources)



def duplicate_list(original_list, n):
	new_list = []
	for item in original_list:
		for i in range(1, n+1):
			new_item = f"{item},{i}"
			new_list.append(new_item)
	return new_list

	
def duplicate_dict(original_dict, n):
	new_dict = {}
	for key, value in original_dict.items():
		for i in range(1, n + 1):
			new_key = f"{key},{i}"
			new_value = copy.deepcopy(value)
			new_dict[new_key] = new_value
	return new_dict



def sersic_fit(ampl, reff, x0, y0, nn, eps, pa, x, y, kernel, mask):
	mod = Sersic2D(amplitude = ampl, r_eff = reff, n=nn, x_0=x0, y_0=y0, ellip=eps, theta =pa)
	img = mod(x, y)
	img_conv = convolve(img, kernel)
	
	return img_conv[mask]


def sersic_plot(ampl, reff, x0, y0, nn, eps, pa, x, y, kernel):
	mod = Sersic2D(amplitude = ampl, r_eff = reff, n=nn, x_0=x0, y_0=y0, ellip=eps, theta =pa)
	img = mod(x, y)
	img_conv = convolve(img, kernel)
	
	return img_conv

def psf_fit(amp, x0, y0, bmaj, bmin, theta, x, y, mask):
	mod = Gaussian2D(amplitude=1.0, x_mean=x0, y_mean=y0, x_stddev=bmaj, y_stddev=bmin, theta=theta, cov_matrix=None)
	img = mod(x, y)*amp
	return img[mask]

def psf_plot(amp, x0, y0, bmaj, bmin, theta, x, y):
	mod = Gaussian2D(amplitude=1.0, x_mean=x0, y_mean=y0, x_stddev=bmaj, y_stddev=bmin, theta=theta, cov_matrix=None)
	img = mod(x, y)*amp
	return img



def inc (e): #ellipticity
	return np.arccos(1-e)*180/np.pi


def fit_sersic_components(pmod, self):
	components = []
	for i in range(1, self.Ncomp + 1):
		sersic_i = sersic_fit(pmod['I$_{e}$'+f',{i}']['value'], pmod['r$_{eff}$'+f',{i}']['value'],
				      pmod[f'x0,{i}']['value'], pmod[f'y0,{i}']['value'],
				      pmod[f'n,{i}']['value'], pmod[f'ϵ,{i}']['value'],
				      pmod[f'PA,{i}']['value'], self.x, self.y,
				      self.kernel, self.mask)
		components.append(sersic_i)

	final_model = sum(components)
	return final_model


def plot_sersic_components(pmod, self):
	components = []
	for i in range(1, self.Ncomp + 1):
		sersic_i = sersic_plot(pmod['I$_{e}$'+f',{i}']['value'], pmod['r$_{eff}$'+f',{i}']['value'],
				      pmod[f'x0,{i}']['value'], pmod[f'y0,{i}']['value'],
				      pmod[f'n,{i}']['value'], pmod[f'ϵ,{i}']['value'],
				      pmod[f'PA,{i}']['value'], self.x, self.y,
				      self.kernel)
		components.append(sersic_i)

	final_model = sum(components)
	return final_model


#Compute surface brightness in rings
def brightness(data, RMAX_arcsec, XPOS, YPOS, inc0, pa0, pix_to_arc, FWHM_arcsec, noise):
	DTR = np.pi/180.
	StN_threshold = 3.

	x, y = np.mgrid[0:data.shape[0],0:data.shape[1]]

	DR = int(np.round(FWHM_arcsec/pix_to_arc)) #distance between rings in pixels

	NRINGS = int(RMAX_arcsec/FWHM_arcsec)+1  #number of ellipses
	radii = np.linspace(0.,RMAX_arcsec,NRINGS) #radii of ellipses in arcsec
	radii_in_px = np.linspace(DR,(NRINGS)*DR,NRINGS) #radii of ellipses in px


	brightness   = np.zeros(NRINGS)
	e_brightness = np.zeros(NRINGS)
	cumulative   = np.zeros(NRINGS-1)
	StN          = np.zeros(NRINGS)
	shape        = (NRINGS, data.shape[0], data.shape[1])
	rings        = np.zeros(NRINGS*data.shape[0]*data.shape[1]).reshape(*shape)

	for n in (range(0,NRINGS-1)):
		e_ext = Ellipse2D(amplitude = 1,
					 x_0 = XPOS,
					 y_0 = YPOS,
					 a = radii_in_px[n+1],
					 b = radii_in_px[n+1]*np.cos(inc0*DTR),
					 theta = np.pi*0.5-pa0)
		#                                 theta = Angle(pa0,'deg').radian)

		e_int = Ellipse2D(amplitude = 1,
					 x_0 = XPOS,
					 y_0 = YPOS,
					 a = radii_in_px[n],
					 b = radii_in_px[n]*np.cos(inc0*DTR),
					 theta = np.pi*0.5-pa0)
		#                                 theta = Angle(pa0,'deg').radian)


		ring  = (e_ext(x,y)-e_int(x,y))
		rings[n] = e_ext(x,y)
		ring[np.where(ring==0)] = np.nan

		values = data*ring
		brightness[n] = np.nanmean(values)


		n_pixels = np.count_nonzero(~np.isnan(values))
		sigma_obs = np.nanstd(values)
		e_brightness[n] = np.max([noise,sigma_obs])/np.sqrt(n_pixels)
		StN[n] = brightness[n]/e_brightness[n]
		if(StN[n]<StN_threshold): break

	return [radii_in_px, brightness, e_brightness, rings]



def make_random_gaussian(par, std, nrandom):
	"""
	make_random_gaussian creates a random gaussian array for an input parameters with a mean value and a standard deviation. This helps with calculating representative errors. We stop the distribution from going to negative values.
	Keyword arguments:
	par     --  mean value of the parameter to compute. (accepted: float or array of floats)
	std     --  standard deviation of the gaussian, float.
	nrandom --  number of elements in the random gaussian array, int.
	:return: array of a gaussian distribution
	"""
	if(len(par)>1):
		array = np.zeros((len(par), nrandom))
		for k, mean in enumerate(par):
			X = stats.truncnorm(a=(0-mean)/std[k], b=1e5, loc=mean, scale=std[k])
			array[k] = X.rvs(nrandom)
	else:
		X = stats.truncnorm(a=(0-par)/std, b=1e5, loc=par, scale=std)
		array = X.rvs(nrandom)
	return array


def polyfit_2nd(x, y, edge):
	"""
	polyfit_2nd fits a 2nd order polynomical to a set of three points and returns the slope at the middle point i.

	Keyword arguments:
	x    --  array with the value of the x coordinates of two or three points. x = [y_i-1, y_i, y_i+1]
	y    --  array with the value of the y coordinates of two or three points. y = [y_i-1, y_i, y_i+1]
	edge --  str, informs if the ith point is at the edge and whether it is left or right. (accepted: left, right or none)
	:return: slope
	"""
	if(edge=='left'):
		x1, x2 = x[0], x[1]
		y1, y2 = y[0], y[1]
		a = (y1 - y2)/(x1-x2)
		b = y2 - a*x2
		slope = a
	if(edge=='right'):
		x0, x1 = x[0], x[1]
		y0, y1 = y[0], y[1]
		a = (y0 - y1)/(x0-x1)
		b = y1 -a*x1
		slope = a
	if(edge=='none'):
		x0, x1, x2 = x[0], x[1], x[2]
		y0, y1, y2 = y[0], y[1], y[2]
		a = ( x0*(y2-y1) + x1*(y0-y2) + x2*(y1-y0) ) / ( (x0-x1) * (x0-x2) * (x1-x2) )
		b = (y1-y0) / (x1-x0) - a*(x0+x1)
		#c = y0 - a*x0*x0 - b*x0
		slope = 2*a*x1+b
	return slope


def ddR_lnsigma(vdisp, radii, nrandom):
	"""
	ddR_lnsigma calculates the derivative of the ln of the velocity dispersion as a function of radius. It returns an array with the slope of the ln sigma at each radial element.
	Keyword arguments:
	vdisp    --  array with the gaussian distribution of the velocity dispersion per radii. Array with nrandom elements per each radii element.
	radii    --  array with the radial points of the galaxy.
	nrandom  --  number of elements in the random gaussian array to compute errors, see function make_random_gaussian
	:return: slope_r, which contains n random elements for each slope at each radii
	"""
	ln_sigma = np.log(vdisp)
	slope_r = np.zeros((len(radii), nrandom))

	k=0
	for k, rad in enumerate(slope_r.T):
		lnvdisp = ln_sigma.T[k]
		myslope = np.zeros(len(radii))
		for r in range(len(radii)):
			if(r==0):
			    slope = polyfit_2nd(x=[radii[r], radii[r+1]], y=[lnvdisp[r], lnvdisp[r+1]], edge='left')
			if(r==len(radii)-1):
			    slope = polyfit_2nd(x=[radii[r-1], radii[r]], y=[lnvdisp[r-1], lnvdisp[r]], edge='right')
			if(r>0 and r<(len(radii)-1)):
			    slope = polyfit_2nd(x=[radii[r-1], radii[r], radii[r+1]], y=[lnvdisp[r-1], lnvdisp[r], lnvdisp[r+1]], edge='none')
			myslope[r] = slope
		slope_r.T[k] = myslope.T
	return slope_r



def VA_square(radii, vdisp, slope, Rgas, nrandom):
	"""
	VA_square calculates the pressure support correction at each radial element (VA). It returns and array with VA**2 per radii.
	Keyword arguments:
	radii    --  array with the radial points of the galaxy.
	vdisp    --  array with the gaussian distribution of the velocity dispersion per radii. Array with nrandom elements per each radii element.
	slope    --  array with the derivative of the ln sigma per radii.
	Rgas     --  array with the gaussian distribution of the Rgas of the gas disc.
	nrandom  --  number of elements in the random gaussian array to compute errors, see function make_random_gaussian
	:return: slope_r, which contains n random elements for each slope at each radii
	"""
	Va_square = np.zeros((len(radii), nrandom))
	for r, rad in enumerate(radii):
		Va_square[r] = -2*(vdisp[r]**2)*rad*slope[r] + rad*(vdisp[r]**2)/Rgas
	return Va_square





def cut_two_images(image1, header1, image2, header2, cut_size = 100):
	wcs1 = WCS(header1, naxis=2)
	wcs2 = WCS(header2, naxis=2)

	# Calculate the center and indices for the cut for image1
	center_y1, center_x1 = image1.shape[0] // 2, image1.shape[1] // 2
	ymin1, ymax1 = center_y1 - cut_size // 2, center_y1 + cut_size // 2
	xmin1, xmax1 = center_x1 - cut_size // 2, center_x1 + cut_size // 2

	center_y2, center_x2 = image2.shape[0] // 2, image2.shape[1] // 2
	ymin2, ymax2 = center_y2 - cut_size // 2, center_y2 + cut_size // 2
	xmin2, xmax2 = center_x2 - cut_size // 2, center_x2 + cut_size // 2

	ext = [xmin1, xmax1, ymin1, ymax1]

	# Slice the image data to the central cut
	image1_cut = image1[ymin1:ymax1, xmin1:xmax1]
	image2_cut = image2[ymin2:ymax2, xmin2:xmax2]


	# Reproject the second image to match the WCS of the first image
	image2_reprojected, footprint = reproject_interp((image2, wcs2), wcs1, shape_out=image1.shape)

	# Slice the reprojected image data to the central cut
	image2_cut = image2_reprojected[ymin1:ymax1, xmin1:xmax1]

	return [image1_cut, image2_cut, wcs1, ext]


def mass_c_relation(MDM, z):
	mdm = 10**MDM

	b = -0.101 + 0.026*z #from Dutton&Maccio14
	a = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z**(1.21))
	MM = (mdm)/1e12

	log10c = a + b*np.log10(MM)

	return(10**log10c)

def V_sersic(R, Mb, Rb, n_ser, arc2kpc):
	# Circular speed of a sersic component
	Mb = 10.0**Mb
	num = G *Mb *u.M_sun

	p_ser = 1.0 -0.6097/n_ser + 0.05463/(n_ser**2) #Terzic, graham 2005, close to eq 4
	b_ser = (2.0*n_ser) -(1./3) + (0.009876/n_ser) #close to eq 2

	z_ser = b_ser*(R/Rb)**(1.0/n_ser)
	gam1 = special.gammainc((3.0-p_ser)*n_ser, z_ser)

	vser = (num.value*gam1)/(R*arc2kpc) #appendix (A1/A2)
	vser = np.sqrt(vser)

	return vser


def V_exp_trunc(R, mm, rd, arc2kpc, rtrunc):
	# Circular speed of an exponential component
	
	mm = 10.0**mm*u.Msun
	Rd = rd*arc2kpc*u.kpc#kpc2m
	Rtrunc = rtrunc*arc2kpc*u.kpc

	y = (R*arc2kpc)/(2.0*Rd.value)

	p1 = special.i0(y)*special.k0(y)
	p2 = special.i1(y)*special.k1(y)


	a = 1 - np.exp(-Rtrunc/Rd)*(1 + Rtrunc/Rd)
	Sigma0 = mm/(2*np.pi*(Rd**2)*a)

	#equation 2.165 Binney and Tremaine 2008
	Vcirc = np.sqrt(4*np.pi*Sigma0*Rd*G*y*y*(p1-p2))


	Vtrunc = np.sqrt(mm*G/Rtrunc)

	return Vcirc.value


def V_exp(R, mm, rd, arc2kpc):
	# Circular speed of an exponential component
	
	mm = 10.0**mm
	rgas = arc2kpc* rd * u.kpc#kpc2m

	y = (R*arc2kpc)/(2.0*rgas.value)

	Vgas = (np.sqrt(2.*G *mm*u.M_sun/rgas)).to(u.km/u.s)

	p1 = special.i0(y)*special.k0(y)
	p2 = special.i1(y)*special.k1(y)

	return Vgas.value*y*np.sqrt(p1-p2)


def Mnfw (V200, Hz):
	# NFW mass from the circular speed at R200
	m200 = (V200**3)/(10.0*const.G.to(u.Mpc*(u.km/u.s)**2/u.M_sun).value*Hz)
	
	return m200

	
def Vnfw (M200, Hz):
	# Circular speed at R200 of an NFW halo with an M200
	v200 = M200 * 10.0 * const.G.to(u.Mpc*(u.km/u.s)**2/u.M_sun).value * Hz
	
	return (v200**(1.0/3))
	
def V_nfw (R, M200, c, Hz, arc2kpc):
	# Circular speed of an NFW halo at a certain radius
	M200 = 10.0**M200
	V200 = Vnfw (M200, Hz) #km/s
	R200 = (V200/(10.0 *Hz))*1e+3 #kpc
	x = (R*arc2kpc)/R200
	num = np.log(1.0 + c * x) - ( c*x )/(1.0 + c*x)
	den = np.log(1.0 + c) - c/(1.0 + c)

	return V200* np.sqrt( num/(den*x))

def V_burkert(R, M200, c, Hz, arc2kpc):
	# Circular speed of a Burkert profile
	M200 = 10.0**M200
	V200 = Vnfw (M200, Hz) #km/s
	R200 = (V200/(10.0 *Hz))*1e+3 #kpc
	x = (R*arc2kpc/R200)*c
	num = 0.5*np.log(1.0 + x * x) + np.log(1.0 + x) - np.arctan(x)
	den = 0.5*np.log(1.0 + c*c) + np.log(1.0 + c)- np.arctan(c)
	
	return V200*np.sqrt(R200/R*arc2kpc)* np.sqrt(num/den)
	

def V_point_source(R, mass, arc2kpc):
	M = 10**mass
	vcirc = np.sqrt(M*G.value/(R*arc2kpc))
	return vcirc


def V_isothermal(R, M200, c, Hz, arc2kpc):
	# Circular speed of an isothermal profile
	M200 = 10.0**M200
	V200 = Vnfw (M200, Hz) #km/s
	R200 = (V200/(10.0 *Hz))*1e+3 #kpc
	x = (R*arc2kpc/R200)*c
	num = 1.0 - np.arctan(x)/x
	den = 1.0 - np.arctan(x)/c
	return V200*np.sqrt(num/den)

def mstarmhalo(mstar): #legrand
	# Coupling stellar mass to halo mass with Legrand's SHMR
	ms = 10.0**mstar
	logm1 =13.14
	beta = 0.631
	m0 = 10.0**11.14
	delta = 0.73
	gamma = 1.0
	
	a1 = logm1 + beta*np.log10(ms/m0)
	a21 = (ms/m0)**delta
	a22 = 1.0+(ms/m0)**(-gamma)
	
	return 10.0**(a1 + a21/a22 -0.5)


def fbaryonmhalo(mstar, alphaco, mgas, fbaryon):
	# Coupling baryon mass to halo mass with a baryonic fraction
	fbar = 10**fbaryon
	mbaryon = 10.0**mstar + 10.0**mgas*alphaco
	mhalo = mbaryon*(1-fbar)/fbar
	return np.log10(mhalo)




def V_sidm(R, rho_s, rs, rc, arc2kpc):
	beta = 4.

	rhos = 10**rho_s

	r = R*arc2kpc
	#from Yang+24, SIDM density profile:
	term = r**beta + rc**beta
	term_a = np.power(term, 0.25)/rs
	term_b = (1. + r/rs)**2.
	rho = rhos / (term_a * term_b)

	#enclosed mass assuming spherical dist:
	mass = 4*np.pi*trapz(rho, dx=0.01)

	vcirc = np.sqrt(mass*G.value/r)

	return vcirc

def BH_host_local(logmstar):
	#ding et al. 2020
	alpha1 = 0.27
	beta1  = 0.98
	mstar = (10**logmstar)/1e10
	MBH = 10**(alpha1 + beta1 * np.log10(mstar)) * 1e7
	return np.log10(MBH)

def BH_host_highz(logmstar, z):
	#ding et al. 2020
	alpha1 = 0.27
	beta1  = 0.98
	mstar = (10**logmstar)/1e10
	MBH = 10**(alpha1 + beta1 * np.log10(mstar)) * 1e7

	gamma = 1.03
	deltaMBH = gamma * np.log10(1+z)

	MBH_highz = np.log10(MBH) + deltaMBH
	logMBH = np.log10(10**MBH_highz/(10**logmstar))

	return logMBH





def plot_dydy(ax, R, Vcirc, errv, vgas, pmod, fg0, fg1, fg2, fg3, fg4, mgas, Hz, arc2kpc, rtruncs, redshift):

	if fg0 == 'point_mass': #black hole
		vbh = V_point_source(R, pmod['mbh']['value'], arc2kpc)
		ax.plot(R*arc2kpc, vbh, color = 'purple', ls= '--', label ='BH')
		bhmass = pmod['mbh']['value']
		print(f'BH mass: {np.round(np.sqrt(bhmass), 2)}')
	elif fg0 == 'bh_host_local':
		bhmass = BH_host_local(pmod['mstar']['value'])
		vbh = V_point_source(R, bhmass, arc2kpc)
		ax.plot(R*arc2kpc, vbh, color = 'purple', ls= '--', label ='BH')
		print(f'BH mass: {np.round(np.sqrt(bhmass), 2)}')
	elif fg0 == 'bh_host_highz':
		bhmass = BH_host_highz(pmod['mstar']['value'], redshift)
		vbh = V_point_source(R, bhmass, arc2kpc)
		ax.plot(R*arc2kpc, vbh, color = 'purple', ls= '--', label ='BH')
		print(f'BH mass: {np.round(np.sqrt(bhmass), 2)}')
	elif fg0 == 'None':
		vbh = 0


	if fg1 == 'sersic':
		vstar = V_sersic(R, pmod['mstar']['value'], pmod['restar']['value'],pmod['n']['value'], arc2kpc)
		vstar_2 = vstar**2
		ax.plot(R*arc2kpc, np.sqrt(vstar_2), color = 'blue',  ls= '-.', label ='Stars, sersic')
	elif fg1 == 'exp':
		vstar = V_exp(R, pmod['mstar']['value'], pmod['restar']['value']/1.678, arc2kpc)
		vstar_2 = vstar**2
		ax.plot(R*arc2kpc, np.sqrt(vstar_2), color = 'blue',  ls= '-.', label ='Stars, exp')
	elif fg1 == 'exp_trunc':
		vstar = V_exp_trunc(R, pmod['mstar']['value'], pmod['restar']['value']/1.678, arc2kpc, rtruncs)
		vstar_2 = vstar**2
		ax.plot(R*arc2kpc, np.sqrt(vstar_2), color = 'blue',  ls= '-.', label ='Stars, exp_trunc')
	elif fg1 == 'bulge_disc':
		stellar_mass = pmod['mstar']['value']
		b_t = pmod['b/t']['value']
		bulge_mass = (10**pmod['mstar']['value'])*pmod['b/t']['value']
		disc_mass = 10**pmod['mstar']['value'] - bulge_mass

		vbulge = V_sersic(R, np.log10(bulge_mass), pmod['restar']['value'],pmod['n']['value'], arc2kpc)
		vdisc = V_exp(R, np.log10(disc_mass), pmod['redisc']['value']/1.678, arc2kpc)
		vstar_2 = vbulge**2 + vdisc**2

		ax.plot(R*arc2kpc, vbulge, color = 'red',  ls= '-.', label ='Bulge')
		ax.plot(R*arc2kpc, vdisc, color = 'blue',  ls= '-.', label ='Stellar Disc')


	elif fg1 == 'exp_psf':
		stellar_mass = pmod['mstar']['value']
		b_t = pmod['b/t']['value']
		bulge_mass = (10**pmod['mstar']['value'])*pmod['b/t']['value']
		disc_mass = 10**pmod['mstar']['value'] - bulge_mass

		vbulge = V_point_source(R, np.log10(bulge_mass), arc2kpc)
		vdisc = V_exp(R, np.log10(disc_mass), pmod['restar']['value']/1.678, arc2kpc)
		vstar_2 = vbulge**2 + vdisc**2

		ax.plot(R*arc2kpc, vbulge, color = 'red',  ls= '-.', label ='Bulge, PSF')
		ax.plot(R*arc2kpc, vdisc, color = 'blue',  ls= '-.', label ='Stellar Disc')

		print(f'bulge mass: {np.round(np.log10(bulge_mass), 2)}')
		print(f'disc mass: {np.round(np.log10(disc_mass), 2)}')


	elif fg1 == 'sersic_psf':
		stellar_mass = pmod['mstar']['value']
		b_t = pmod['b/t']['value']
		bulge_mass = (10**pmod['mstar']['value'])*b_t
		disc_mass = 10**pmod['mstar']['value'] - bulge_mass

		vbulge = V_point_source(R, np.log10(bulge_mass), arc2kpc)
		vdisc = V_sersic(R, np.log10(disc_mass), pmod['restar']['value'], pmod['n']['value'], arc2kpc)
		vstar_2 = vbulge**2 + vdisc**2

		ax.plot(R*arc2kpc, vbulge, color = 'red',  ls= '-.', label ='Bulge, PSF')
		ax.plot(R*arc2kpc, vdisc, color = 'blue',  ls= '-.', label ='Stellar Disc')

		print(f'bulge mass: {np.round(np.log10(bulge_mass), 2)}')
		print(f'disc mass: {np.round(np.log10(disc_mass), 2)}')



	elif fg1 == 'double_exp':
		stellar_mass = pmod['mstar']['value']
		d_t = pmod['d1/t']['value']
		disc1_mass = (10**pmod['mstar']['value'])*d_t
		disc2_mass = (10**(pmod['mstar']['value'])) - disc1_mass

		vdisc1 = V_exp(R, np.log10(disc1_mass), pmod['restar']['value']/1.678, arc2kpc)
		vdisc2 = V_exp(R, np.log10(disc2_mass), pmod['restar2']['value']/1.678, arc2kpc)
		vstar_2 = vdisc1**2 + vdisc2**2

		ax.plot(R*arc2kpc, vdisc1, color = 'darkblue',  ls= '-.', label ='Stellar Disc 1')
		ax.plot(R*arc2kpc, vdisc2, color = 'blue',  ls= '-.', label ='Stellar Disc 2')

		print(f'disc1 mass: {np.round(np.log10(disc1_mass), 2)}')
		print(f'disc2 mass: {np.round(np.log10(disc2_mass), 2)}')


	if fg2 == 'nfw':
		vdm = V_nfw(R, pmod['mdm']['value'], pmod['cdm']['value'], Hz, arc2kpc)
	elif fg2 == 'nfw_c':
		C200 = mass_c_relation(pmod['mdm']['value'], redshift)
		vdm = V_nfw(R, pmod['mdm']['value'], C200, Hz, arc2kpc)
		print('mdm and concentration:', np.round(pmod['mdm']['value'],2), np.round(C200,2))
	elif fg2 == 'burkert':
		vdm = V_burkert(R, pmod['mdm']['value'], pmod['cdm']['value'], Hz, arc2kpc)
	elif fg2 == 'isothermal':
		vdm = V_isothermal(R, pmod['mdm']['value'], pmod['cdm']['value'], Hz, arc2kpc)
	elif fg2 == 'nodm':
		vdm = 0
	elif fg2 == 'sidm':
		vdm = V_sidm(R, pmod['rho_s']['value'], pmod['rs']['value'], pmod['rc']['value'], arc2kpc)

	if fg2 == 'fbar':
		DMmass = fbaryonmhalo(pmod['mstar']['value'],pmod['alphaco']['value'],mgas, pmod['fbar']['value'])
		print('M200: ', np.round(DMmass,2))
		print('fbar: ', np.round(10**pmod['fbar']['value'], 2))
		vdm = V_nfw(R, DMmass, pmod['cdm']['value'], Hz, arc2kpc)
	if fg2 == 'fbar_cmr':
		DMmass = fbaryonmhalo(pmod['mstar']['value'],pmod['alphaco']['value'],mgas, pmod['fbar']['value'])
		C200 = mass_c_relation(DMmass, redshift)
		print('M200: ', np.round(DMmass,2))
		print('C200: ', np.round(C200,2))
		print('fbar: ', np.round(10**pmod['fbar']['value'], 2))
		vdm = V_nfw(R, DMmass, C200, Hz, arc2kpc)

	if   fg4 == 'const':
		vgas_2 = vgas**2*pmod['alphaco']['value']
		print('Mgas: ', np.round(np.log10((10**mgas)*pmod['alphaco']['value']),2))


	model = np.sqrt(vstar_2 + vgas_2 + vdm**2 + vbh**2)


	
	ax.errorbar(R*arc2kpc, Vcirc, yerr =  errv, fmt ='o')
		
	ax.plot(R*arc2kpc, np.sqrt(vgas_2), color = 'orange', ls= '--', label ='Gas')

	print('############################')


	if fg1 == 'exp_psf' or fg1 == 'bulge_disc' or fg1 == 'sersic_psf':
		print(f'vbulge: {np.round(vbulge, 2)}')
		print(f'vdisc: {np.round(vdisc, 2)}')
	elif fg1 == 'double_exp':
		print(f'vdisc1: {np.round(vdisc1, 2)}')
		print(f'vdisc2: {np.round(vdisc2, 2)}')
	elif fg1 == 'sersic' or fg1 == 'exp' or fg1 == 'exp_trunc':
		print(f'vstar: {np.round(np.sqrt(vstar_2), 2)}')

	print('vgas:', np.round(np.sqrt(vgas_2),2))

	if fg0 != 'None' : print(f'vbh: {np.round(np.sqrt(vbh_2), 2)}')
	if fg2!= 'nodm':
		ax.plot(R*arc2kpc, vdm, color = 'green', label = 'DM')



	ax.plot(R*arc2kpc, model, color ='black', label ='Total')
	ax.set_xlabel('Radius (kpc)')
	ax.set_ylabel('V$_{\mathrm{circ}}$ (km s$^{-1}$)')
	ax.set_xlim(0, R[-1]*arc2kpc*1.1)
	ax.set_ylim(0, np.amax(model)*1.2)


	ax.legend()
	#print(R, vstar, vgas)
	#print(R, vstar/vgas)
	print('vdm: ', np.round(vdm,2))
	print('R: ', np.round(R*arc2kpc, 3))

"""
See his slack messages: 
 - https://darkenergysurvey.slack.com/archives/C068GFSR2F5/p1731721322661459
 - https://darkenergysurvey.slack.com/archives/C068GFSR2F5/p1731721942226649
"""
import numpy as np


class Tz:
    def __init__(self, dz, nz, z0=None):
        '''Class representing sawtooth n(z) kernels (bins) in z.
        First kernel is centered at z0, which defaults to dz if not
        given.  If z0<dz, then its triangle is adjusted to go to zero at
        0, then peak at z0, down to zero at z0+dz/2.
        Arguments:
        `dz`: the step between kernel centers
        `nz`: the number of kernels.
        `z0`: peak of first bin'''
        self.dz = dz
        self.nz = nz
        if z0 is None:
            self.z0 = dz
        else:
            self.z0 = z0
        # Set a flag if we need to treat kernel 0 differently
        self.cut0 = self.z0<dz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.

        if self.cut0 and k==0:
            # Lowest bin is unusual:
            out = jnp.where(z>self.z0, 1-(z-self.z0)/self.dz, z/self.z0)
            out = jnp.maximum(0., out) / ((self.z0+self.dz)/2.)
        else:
            out = jnp.maximum(0., 1 - jnp.abs((z-self.z0)/self.dz-k)) / self.dz
        return out
    
    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmax = np.arange(1,1+self.nz)*self.dz + self.z1
        zmin = zmax = 2*self.dz
        if self.cut0:
            zmin[0] = 0.
        return np.stack( (zmin, zmax), axis=1)
    
    def dndz(self,coeffs, z):
        '''Calculate dn/dz at an array of z values given set(s) of
        coefficients for the kernels/bins.  The coefficients will
        be normalized to sum to unity, i.e. they will represent the
        fractions within each kernel.
        Arguments:
        `coeffs`:  Array of kernel fractions of shape [...,nz]
        `z`:       Array of redshifts of arbitrary length
        Returns:
        Array of shape [...,len(z)] giving dn/dz at each z for
        each set of coefficients.'''
        
        # Make the kernel coefficients at the z's
        kk = np.array([self(k,z) for k in range(self.nz)])
        return np.einsum('...i,ij->...j',coeffs,kk) / np.sum(coeffs, axis=-1)

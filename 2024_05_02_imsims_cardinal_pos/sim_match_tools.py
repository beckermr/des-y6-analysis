import numpy as np
import galsim as gs
import hpgeom


def project_to_tile(ra, dec, cen_ra, cen_dec, tile_wcs):
    # build the DES-style coadd WCS at center ra,dec
    aft = gs.AffineTransform(
        -0.263,
        0.0,
        0.0,
        0.263,
        origin=gs.PositionD(5000.5, 5000.5),
    )
    cen = gs.CelestialCoord(ra=cen_ra * gs.degrees, dec=cen_dec * gs.degrees)
    cen_wcs = gs.TanWCS(aft, cen, units=gs.arcsec)    

    # project to u,v about cen - needs radians as input
    u, v = cen_wcs.center.project_rad(
        np.radians(ra),
        np.radians(dec)
    )

    # now deproject about tile_wcs - comes out in radians
    new_ra, new_dec = tile_wcs.center.deproject_rad(u, v)
    new_ra, new_dec = np.degrees(new_ra), np.degrees(new_dec)
    x, y = tile_wcs.radecToxy(new_ra, new_dec, units="degrees")
    
    return new_ra, new_dec, x, y


def sample_from_pixel(nside, pix, size=None, nest=True, rng=None):
    if rng is None:
        rng = np.random.RandomState(seed=rng)

    if size == None:
        ntot = 1
        scalar = True
    else:
        ntot = np.prod(size)
        scalar = False
    
    ra, dec = hpgeom.pixel_to_angle(
        nside,
        pix,
        nest=nest,
    )
    ddec = 2.0 * hpgeom.nside_to_resolution(nside)
    dec_range = np.array([
        dec - ddec,
        dec + ddec,
    ])
    dec_range = np.clip(dec_range, -90, 90)
    cosdec = np.cos(np.radians(dec))
    dra = ddec / cosdec
    ra_range = np.array([
        ra - dra,
        ra + dra
    ])
    sin_dec_range = np.sin(np.radians(dec_range))
    
    ra = np.empty(ntot)
    dec = np.empty(ntot)
    nsofar = 0
    while nsofar < ntot:
        ngen = int(1.5 * min(ntot, ntot - nsofar))
        _ra = rng.uniform(low=ra_range[0], high=ra_range[1], size=ngen)
        _sindec = rng.uniform(low=sin_dec_range[0], high=sin_dec_range[1], size=ngen)
        _dec = np.degrees(np.arcsin(_sindec))
        inds = np.where(hpgeom.angle_to_pixel(nside, _ra, _dec, nest=nest) == pix)[0]
        _n = inds.shape[0]
        _n = min(_n, ntot - nsofar)
        if _n > 0:
            ra[nsofar:nsofar+_n] = _ra[inds[:_n]]
            dec[nsofar:nsofar+_n] = _dec[inds[:_n]]
            nsofar += _n
            
    if scalar:
        return ra[0], dec[0]
    else:
        return np.reshape(ra, size), np.reshape(dec, size)

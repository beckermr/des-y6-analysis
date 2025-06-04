import numpy as np
import galsim as gs
import hpgeom
from scipy.spatial import KDTree
from numba import njit
from esutil.pbar import PBar


@njit
def _get_smallest_dist(dist, inds, allowed_inds):
    min_dist = np.inf
    min_ind = -1
    for allowed_ind in allowed_inds:
        for j in range(inds.shape[0]):
            if allowed_ind == inds[j]:
                break

        if dist[j] < min_dist:
            min_dist = dist[j]
            min_ind = allowed_ind

    return min_ind


def match_cosmos_to_cardinal(cosmos, cardinal, photo_scale_factor=3, max_radius=0.3):
    # we cut the cosmos redshift range to match cardinal
    match_inds = np.zeros(len(cosmos), dtype=int) - 1
    msk_cosmos_to_match = (
        cosmos["photoz"] < 2.3
    )
    inds_msk_cosmos_to_match = np.where(msk_cosmos_to_match)[0]
    mcosmos = cosmos[msk_cosmos_to_match]

    # we remove very bright things from cardinal
    msk_cardinal_to_match_to = (
        cardinal["TMAG"][:, 2] > 17.5
    )
    inds_msk_cardinal_to_match_to = np.where(msk_cardinal_to_match_to)[0]
    cardinal = cardinal[msk_cardinal_to_match_to]

    # we match in i-band, g-i color, and photo-z
    # we scale the photo-z by 3 to match the typical range of magnitudes
    # this has the side effect of equating differences of 0.1 in photo-z to 0.3
    # in magnitudes or colors, which is about right
    cd_data = np.zeros((cardinal.shape[0], 3))
    cd_data[:, 0] = cardinal["TMAG"][:, 2]  # i-band
    cd_data[:, 1] = cardinal["TMAG"][:, 0] - cardinal["TMAG"][:, 2]  # g-i color
    cd_data[:, 2] = cardinal["Z"] * photo_scale_factor

    cs_data = np.zeros((mcosmos.shape[0], 3))
    cs_data[:, 0] = mcosmos["mag_i_dered"]
    cs_data[:, 1] = mcosmos["mag_g_dered"] - mcosmos["mag_i_dered"]
    cs_data[:, 2] = mcosmos["photoz"] * photo_scale_factor

    print("made catalogs", flush=True)

    # now find the closest nbrs in cardinal to each cosmos object
    tree = KDTree(cd_data)
    print("made tree", flush=True)

    nbr_inds = tree.query_ball_point(
        cs_data,
        max_radius,
        eps=0,
        return_sorted=True,
    )
    print("did radius search", flush=True)

    # do brightest things first
    binds = np.argsort(mcosmos["mag_i_dered"])

    # now we loop over the cosmos objects and find the closest cardinal object
    # that has not been matched yet
    used_cd_inds = set()
    for i in PBar(binds, desc="getting best matches"):
        allowed_inds = [ind for ind in nbr_inds[i] if ind not in used_cd_inds and ind < cd_data.shape[0]]
        if allowed_inds:
            min_ind = allowed_inds[0]
            used_cd_inds.add(min_ind)
            match_inds[inds_msk_cosmos_to_match[i]] = inds_msk_cardinal_to_match_to[min_ind]

    print("found matches for %0.2f percent of cosmos" % (np.mean(match_inds >= 0) * 100), flush=True)

    return match_inds


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
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(seed=rng)

    if size is None:
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

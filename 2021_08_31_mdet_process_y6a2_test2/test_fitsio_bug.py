import numpy as np
import esutil as eu
import scipy
import ngmix
import meds
import piff
import fitsio
import desmeds
import pixmappy
from meds.maker import MEDS_FMT_VERSION
from pizza_cutter import __version__
import os
import yaml
import json
import copy

MAGZP_REF = 30


def _build_metadata(*, config, json_info):
    numpy_version = np.__version__
    scipy_version = scipy.__version__
    esutil_version = eu.__version__
    fitsio_version = fitsio.__version__
    meds_version = meds.__version__
    piff_version = piff.__version__
    pixmappy_version = pixmappy.__version__
    desmeds_version = desmeds.__version__
    ngmix_version = ngmix.__version__
    dt = [
        ('magzp_ref', 'f8'),
        ('config', 'S%d' % len(config)),
        ('tile_info', 'S%d' % len(json_info)),
        ('pizza_cutter_version', 'S%d' % len(__version__)),
        ('numpy_version', 'S%d' % len(numpy_version)),
        ('scipy_version', 'S%d' % len(scipy_version)),
        ('esutil_version', 'S%d' % len(esutil_version)),
        ('ngmix_version', 'S%d' % len(ngmix_version)),
        ('fitsio_version', 'S%d' % len(fitsio_version)),
        ('piff_version', 'S%d' % len(piff_version)),
        ('pixmappy_version', 'S%d' % len(pixmappy_version)),
        ('desmeds_version', 'S%d' % len(desmeds_version)),
        ('meds_version', 'S%d' % len(meds_version)),
        ('meds_fmt_version', 'S%d' % len(MEDS_FMT_VERSION)),
        ('meds_dir', 'S%d' % len(os.environ['MEDS_DIR'])),
        ('piff_data_dir', 'S%d' % len(os.environ.get('PIFF_DATA_DIR', ' '))),
        ('desdata', 'S%d' % len(os.environ.get('DESDATA', ' ')))]
    metadata = np.zeros(1, dt)
    metadata['magzp_ref'] = MAGZP_REF
    metadata['config'] = config
    metadata['numpy_version'] = numpy_version
    metadata['scipy_version'] = scipy_version
    metadata['esutil_version'] = esutil_version
    metadata['ngmix_version'] = ngmix_version
    metadata['fitsio_version'] = fitsio_version
    metadata['piff_version'] = piff_version
    metadata['pixmappy_version'] = pixmappy_version
    metadata['desmeds_version'] = desmeds_version
    metadata['meds_version'] = meds_version
    metadata['meds_fmt_version'] = MEDS_FMT_VERSION
    metadata['pizza_cutter_version'] = __version__
    metadata['meds_dir'] = os.environ['MEDS_DIR']
    metadata['piff_data_dir'] = os.environ.get('PIFF_DATA_DIR', '')
    metadata['desdata'] = os.environ.get('DESDATA', '')
    metadata['tile_info'] = json_info
    return metadata


with open("des-pizza-slices-y6-v9.yaml", 'r') as fp:
    _config = fp.read()

info = os.path.expandvars(
    "${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/"
    "DES2132-5748_r_pizza_cutter_info.yaml"
)
with open(info, 'r') as fp:
    info = yaml.load(fp, Loader=yaml.Loader)

# this loads all of the data we need into the info dict
json_info = json.dumps(copy.deepcopy(info))

assert len(_config) == len(_config.encode())
assert len(json_info) == len(json_info.encode())

md = _build_metadata(config=_config, json_info=json_info)


with fitsio.FITS("test.fits", "rw", clobber=True) as fits:
    fits.write(md, extname="metadata")

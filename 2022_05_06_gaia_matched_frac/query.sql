select
  d.ext_mash,
  g.PHOT_G_MEAN_MAG,
  d.MAG_AUTO_G,
  d.MAG_AUTO_R,
  d.MAG_AUTO_I,
  d.MAG_AUTO_Z,
  d.COADD_OBJECT_ID,
  g.SOURCE_ID,
  d.ra,
  d.dec
from
  GAIA_DR2_X_Y6_GOLD_2_0 m,
  GAIA_DR2 g,
  Y6_GOLD_2_0 d
where
  m.COADD_OBJECT_ID = d.COADD_OBJECT_ID
  and m.SOURCE_ID = g.SOURCE_ID
  and d.flags_footprint > 0
  and d.flags_gold = 0
  and d.flags_foreground = 0; > gaia.fits

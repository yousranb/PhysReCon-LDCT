"""PhysReCon-LDCT: physics-grounded multiscale degradation learning for few-shot adaptive LDCT synthesis.

Modules
  physics   shared acquisition physics (projection, incremental noise, reference FBP)   [Module A]
  wavelet   undecimated Haar decomposition with exact inverse
  stats     tissue masks, flat ROIs, NPS, degradation descriptor, covariances, SWD, edge profiles
  model     reconstruction-response CNN (Module C), embedding (Module B), signal transfer (Module D)
  losses    distributional / statistical / signal-transfer objectives (no NDCT target)
  data      domain keys, splits, support selection
"""

from arosics import COREG_LOCAL

im_reference = r''
im_target    = r''
kwargs = {
    'grid_res'     : 200,
    'window_size'  : (256, 256),
    'path_out'     : 'auto',
    'projectDir'   : 'coregistration',
    'q'            : False,
}

CRL = COREG_LOCAL(im_reference,im_target,**kwargs)
CRL.correct_shifts()
'''

    Error based functions

    jwaiton 240426
'''

import numpy as np


def ratio_error(f, a, b, a_error, b_error):
    '''
    docs for this online, need to move them over
    '''

    return f*np.sqrt((a_error/a)**2 + (b_error/b)**2)


def fom_error(a, b, a_error, b_error):
    '''
    docs for this online, move them over
    derived in joplin notes 11/04/24
    '''

    element_1 = np.square(a_error/np.sqrt(b))
    element_2 = np.square((b_error * a) /(2*(b**(3/2))))
    return np.sqrt(element_1 + element_2)


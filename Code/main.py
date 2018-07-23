# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy.stats import norm
import seaborn

NELEMENTS = 101

SIGMA = 1.0
FATAL = 0.9
BOUNDARY = 0.1

TOTALELEMENTS = int(NELEMENTS * (1+2*BOUNDARY) )

MAXSTEPS = 100

def createCDF( sigma1, value ):
    rv = norm( 0.0, sigma1 )
    x = np.linspace( -1 - BOUNDARY, 1 + BOUNDARY, TOTALELEMENTS)
    y = rv.pdf(x)
    print('sum pdf', sum(y) )
    #thres = 1 - norm.cdf(fatal)
    #print('thres', thres )
    lim1 = int(BOUNDARY * NELEMENTS)
    lim2 = int( TOTALELEMENTS - 1 - int(BOUNDARY*NELEMENTS) )
    print('NELEMENTS', NELEMENTS, 'TOTALELEMENTS', TOTALELEMENTS, 'lim1', lim1, 'lim2', lim2)
    y[0:lim1] = value/(2*BOUNDARY)
    y[lim2:] = value/(2*BOUNDARY)
    print(rv.cdf(-1), rv.cdf(1), rv.cdf(1) - rv.cdf(-1) )
    y = y/np.sum(y)
    return x,y

def drawFromPDF( x, pdf ):
    u = np.random.random()
    sum = 0.0
    for i in range(len(pdf) + 1 ):
        sum = sum + pdf[i]
        if ( i >= len(pdf ) or sum >= u ):
            ind = i
            print('u', u, 'ind', ind, 'x[ind]', x[ind] )
            break
    return x[ind]

def experiment( x, y ):
    disp = 0.0
    for i in range(MAXSTEPS):
        z = drawFromPDF(x,y)
        print('experiment', 'disp', disp, 'z', z )
        if ( z <= -1 ) or ( z >= 1 ):
            break
        disp = disp + z
        if ( disp <= -1 ) or ( disp >= 1 ):
            break
    print('experiment returns', i)
    return i


sigmas = [ 1.0, 1.0, 0.5, 1.0]
values = [ 0.05, 0.01, 0.05, 0.06 ]

def main( argv = None ):
    if argv is None:
        argv = sys.argv
    fig = plt.figure()
    for i in range(4):
        x, y = createCDF(sigmas[i], values[i])
        ax = fig.add_subplot(2,2,i+1)
        ax.set_xlim((-1 - BOUNDARY, 1 + BOUNDARY))
        ax.set_ylim((0, 0.1))
        ax.plot(x,y,label='σ2={0}, f={1}'.format(sigmas[i], values[i]))
        ax.legend()
    print(sum(y))
    fig.savefig('pdf2.png')
    plt.show()

    fig = plt.figure()
    for i in range(4):
        ax=fig.add_subplot(2,2,i+1)
        x, y = createCDF(sigmas[i], values[i])

        count = []
        for e in range(1000):
            t = experiment(x, y)
            count = count + [t]
        ax.hist(count, label='σ2={0}, f={1}'.format(sigmas[i], values[i]))
        m = max(count)
        ax.plot([m,m], [0,1000],'b-', linewidth=3, label='max={0}'.format(m))
        ax.set_xlim((0,50))
        ax.set_ylim((0,500))
        ax.legend()

    fig.savefig( 'hist2.png')
    plt.show()

if __name__ == '__main__':
    main()

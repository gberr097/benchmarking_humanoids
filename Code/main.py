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

np.random.seed(42)

# def createCDF( sigma1, value ):
#     rv = norm( 0.0, sigma1 )
#     x = np.linspace( -1 - BOUNDARY, 1 + BOUNDARY, TOTALELEMENTS)
#     y = rv.pdf(x)
#     print('sum pdf', sum(y) )
#     #thres = 1 - norm.cdf(fatal)
#     #print('thres', thres )
#     lim1 = int(BOUNDARY * NELEMENTS)
#     lim2 = int( TOTALELEMENTS - 1 - int(BOUNDARY*NELEMENTS) )
#     print('NELEMENTS', NELEMENTS, 'TOTALELEMENTS', TOTALELEMENTS, 'lim1', lim1, 'lim2', lim2)
#     y[0:lim1] = value/(2*BOUNDARY)
#     y[lim2:] = value/(2*BOUNDARY)
#     print(rv.cdf(-1), rv.cdf(1), rv.cdf(1) - rv.cdf(-1) )
#     y = y/np.sum(y)
#     return x,y

def createCDF( sigma1, value, nelements, boundary ):
    rv = norm( 0.0, sigma1 )
    x = np.linspace( -1 - boundary, 1 + boundary, int( nelements + 2*0.1 * nelements ) )
    y = rv.pdf(x)
    #thres = 1 - norm.cdf(fatal)
    #print('thres', thres )
    lim1 = int(boundary * nelements)
    lim2 = int( len(x) - 1 - int(boundary * nelements) )
    #weight = norm.cdf(1) - norm.cdf(-1)
    weight = sum(y[lim1:lim2])
    print('w', weight)
    y = y/weight * (1-value)
    print('sum(y)', sum(y), 'sum2', sum(y[lim1:lim2]) )
    weightB = 2*boundary
    y[0:lim1] = value/(2*lim1)
    y[lim2:] = value/(2*lim1)
    print('sum[0:{0}]={1}, sum[{0}:{2}]={3}, sum[{2}:{4}={5}, sum[0:{4}]={6}'.format(lim1, sum(y[0:lim1]), lim2, sum(y[lim1:lim2]), len(y), sum(y[lim2:]), sum(y[0:])))
    return x,y

def drawFromPDF( x, pdf ):
    u = np.random.random()
    sum = 0.0
    for i in range(len(pdf) + 1 ):
        sum = sum + pdf[i]
        if ( i >= len(pdf ) or sum >= u ):
            ind = i
            #print('u', u, 'ind', ind, 'x[ind]', x[ind] )
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

def drawPDF( sigmas, values ):
    fig = plt.figure()
    for i in range(4):
        x, y = createCDF(sigmas[i], values[i], 100, 0.1)
        ax = fig.add_subplot(2,2,i+1)
        #ax.set_title('Model M')
        ax.set_xlim((-1 - BOUNDARY, 1 + BOUNDARY))
        ax.set_ylim((0, 0.1))
        ax.plot(x,y,label='σ2={0}, e_c={1}'.format(sigmas[i], values[i]), linewidth=3,)
        ax.legend()
    print(sum(y))
    fig.savefig('pdf2.eps')
    plt.show()

def drawHist( sigmas, values, NEXPERIMENTS=1000 ):
    fig = plt.figure()
    for i in range(4):
        ax=fig.add_subplot(2,2,i+1)
        x, y = createCDF(sigmas[i], values[i],100, 0.1)

        count = []
        for e in range(NEXPERIMENTS):
            t = experiment(x, y)
            count = count + [t]
        counts, bins, patches = ax.hist(count, 10, label='σ2={0}, e_c={1:5.2f}'.format(sigmas[i], values[i]) )

        print('sum', np.sum(counts), 'counts', counts, 'sum', np.sum(bins), 'bins', bins)
        sum = 0
        for j in range(len(counts)):
            sum = sum + counts[j]
            if sum > 0.95 * NEXPERIMENTS:
                p95 = j
                break
        m = max(count)
        ax.plot([m,m], [0,1000],'b-', linewidth=3, label='max={0:5.2f}'.format(m))
        ax.plot([bins[p95],bins[p95]],[0,1000], 'r--', linewidth=3, label='95%={0:5.2f}'.format(bins[p95]))
        ax.set_xlim((0,100))
        ax.set_ylim((0,500))
        ax.legend()

    fig.savefig( 'hist2.eps')
    plt.show()

def drawExperiment( sigma, value, steps ):
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    x, y = createCDF(sigma, value, 100, 0.1)

    logs = [ [], [], []  ]

    while len(logs[0]) <= 0 or len(logs[1]) <= 0 or len(logs[2]) <= 0:
        print( len(logs[0]), len(logs[1]), len(logs[2]) )
        log = [0]
        disp = 0
        for i in range( steps ):
            z = drawFromPDF(x, y)
            if (z <= -1) or (z >= 1):
                if ( len(log) > steps//2):
                    if ( disp < 0 ):
                        z = +1
                    else:
                        z = -1
                    log.append( z )
                    logs[1] = log
                break
            disp = disp + z
            if (disp <= -1) or (disp >= 1):
                if ( len(log) > steps//2):
                    log.append(disp)
                    logs[0] = log
                break
            log.append( disp )
        if len(log) >= steps:
            logs[2] = log

    ax.plot( logs[0], 'g--', linewidth=3, label='Drift Error' )
    ax.plot(logs[1], 'r-*', linewidth=3, label='Critical Error')
    ax.plot(logs[2], 'b-', linewidth=3, label='No Error')
    ax.set_title( 'Trace of Experiments with σ2 = {0} and e_c = {1}'.format(sigma, value) )
    ax.legend()

    fig.savefig( 'plot2.eps')
    plt.show()

def main( argv = None ):
    if argv is None:
        argv = sys.argv

    sigmas = [ 0.5, 0.5, 1.0, 0.01 ]
    values = [ 0.02, 0.01, 0.001, 0.01 ]

    drawPDF( sigmas, values )

    drawHist( sigmas, values )
    #drawExperiment( 2.0, 0.03, 25 )

if __name__ == '__main__':
    main()

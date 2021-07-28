import datetime
import os
import time
from optparse import OptionParser
from sys import argv, stdout

import nibabel as nib
import numpy as np
import scipy.stats as stats
from numpy import *
from numpy import mean as mean
from numpy import round as round
from numpy import vstack as vstack
from numpy import zeros as zeros
from numpy.linalg import norm
from numpy.linalg import norm as norm
from numpy.linalg import qr as qr
from numpy.random import randn as randn
from scipy.linalg import qr
from scipy.sparse.linalg import svds as svds

__version__='0.1'


def eimask(dd,ees=None):
    if ees==None: ees=range(dd.shape[1])
    imask = np.zeros([dd.shape[0],len(ees)])
    for ee in ees:
        print ee
        lthr = 0.001*scoreatpercentile(dd[:,ee,:].flatten(),98)
        hthr = 5*scoreatpercentile(dd[:,ee,:].flatten(),98)
        print lthr,hthr
        imask[dd[:,ee,:].mean(1) > lthr,ee]=1
        imask[dd[:,ee,:].mean(1) > hthr,ee]=0
    return imask


def scoreatpercentile(a, per, limit=(), interpolation_method='lower'):
    """
    This function is grabbed from scipy

    """
    values = np.sort(a, axis=0)
    if limit:
        values = values[(limit[0] <= values) & (values <= limit[1])]

    idx = per /100. * (values.shape[0] - 1)
    if (idx % 1 == 0):
        score = values[idx]
    else:
        if interpolation_method == 'fraction':
            score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                 idx % 1)
        elif interpolation_method == 'lower':
            score = values[np.floor(idx)]
        elif interpolation_method == 'higher':
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction', " \
                             "'lower' or 'higher'")
    return score


def niwrite(data,affine, name , header=None):
    stdout.write(" + Writing file: %s ...." % name)

    thishead = header
    if thishead == None:
        thishead = head.copy()
        thishead.set_data_shape(list(data.shape))

    outni = nib.Nifti1Image(data,affine,header=thishead)
    outni.to_filename(name)
    print 'done.'

def cat2echos(data,Ne):
    """
    cat2echos(data,Ne)

    Input:
    data shape is (nx,ny,Ne*nz,nt)
    """
    nx,ny = data.shape[0:2]
    nz = data.shape[2]/Ne
    if len(data.shape) >3:
        nt = data.shape[3]
    else:
        nt = 1
    return np.reshape(data,(nx,ny,nz,Ne,nt),order='F')

def uncat2echos(data,Ne):
    """
    uncat2echos(data,Ne)

    Input:
    data shape is (nx,ny,Ne,nz,nt)
    """
    nx,ny = data.shape[0:2]
    nz = data.shape[2]*Ne
    if len(data.shape) >4:
        nt = data.shape[4]
    else:
        nt = 1
    return np.reshape(data,(nx,ny,nz,nt),order='F')

def makemask(cdat):

    nx,ny,nz,Ne,nt = cdat.shape

    mask = np.ones((nx,ny,nz),dtype=np.bool)

    for i in range(Ne):
        tmpmask = (cdat[:,:,:,i,:] != 0).prod(axis=-1,dtype=np.bool)
        mask = mask & tmpmask

    return mask

def fmask(data,mask):
    """
    fmask(data,mask)

    Input:
    data shape is (nx,ny,nz,...)
    mask shape is (nx,ny,nz)

    Output:
    out shape is (Nm,...)
    """

    s = data.shape
    sm = mask.shape

    N = s[0]*s[1]*s[2]
    news = []
    news.append(N)

    if len(s) >3:
        news.extend(s[3:])

    tmp1 = np.reshape(data,news)
    fdata = tmp1.compress((mask > 0 ).ravel(),axis=0)

    return fdata.squeeze()

def unmask (data,mask):
    """
    unmask (data,mask)

    Input:

    data has shape (Nm,nt)
    mask has shape (nx,ny,nz)

    """
    M = (mask != 0).ravel()
    Nm = M.sum()

    nx,ny,nz = mask.shape

    if len(data.shape) > 1:
        nt = data.shape[1]
    else:
        nt = 1

    out = np.zeros((nx*ny*nz,nt),dtype=data.dtype)
    out[M,:] = np.reshape(data,(Nm,nt))

    return np.reshape(out,(nx,ny,nz,nt))

def t2smap(catd,mask,tes):
    """
    t2smap(catd,mask,tes)

    Input:

    catd  has shape (nx,ny,nz,Ne,nt)
    mask  has shape (nx,ny,nz)
    tes   is a 1d numpy array
    """
    nx,ny,nz,Ne,nt = catd.shape
    N = nx*ny*nz

    echodata = fmask(catd,mask)
    Nm = echodata.shape[0]

    #Do Log Linear fit
    B = np.reshape(np.abs(echodata), (Nm,Ne*nt)).transpose()
    B = np.log(B)
    x = np.array([np.ones(Ne),-tes])
    X = np.tile(x,(1,nt))
    X = np.sort(X)[:,::-1].transpose()

    beta,res,rank,sing = np.linalg.lstsq(X,B)
    t2s = 1/beta[1,:].transpose()
    s0  = np.exp(beta[0,:]).transpose()

    #Goodness of fit
    alpha = (np.abs(B)**2).sum(axis=0)
    t2s_fit = blah = (alpha - res)/(2*res)

    out = unmask(t2s,mask),unmask(s0,mask),unmask(t2s_fit,mask)

    return out

def get_coeffs(data,mask,X,add_const=False):
    """
    get_coeffs(data,X)

    Input:

    data has shape (nx,ny,nz,nt)
    mask has shape (nx,ny,nz)
    X    has shape (nt,nc)

    Output:

    out  has shape (nx,ny,nz,nc)
    """
    mdata = fmask(data,mask).transpose()

    X=np.atleast_2d(X)
    if X.shape[0]==1: X=X.T
    Xones = np.atleast_2d(np.ones(np.min(mdata.shape))).T
    if add_const: X = np.hstack([X,Xones])

    tmpbetas = np.linalg.lstsq(X,mdata)[0].transpose()
    if add_const: tmpbetas = tmpbetas[:,:-1]
    out = unmask(tmpbetas,mask)

    return out


def andb(arrs):
    result = np.zeros(arrs[0].shape)
    for aa in arrs: result+=np.array(aa,dtype=np.int)
    return result

def optcom(data,t2s,tes,mask):
    """
    out = optcom(data,t2s)


    Input:

    data.shape = (nx,ny,nz,Ne,Nt)
    t2s.shape  = (nx,ny,nz)
    tes.shape  = (Ne,)

    Output:

    out.shape = (nx,ny,nz,Nt)
    """
    nx,ny,nz,Ne,Nt = data.shape

    fdat = fmask(data,mask)
    ft2s = fmask(t2s,mask)

    tes = tes[np.newaxis,:]
    ft2s = ft2s[:,np.newaxis]

    alpha = tes * np.exp(-tes /ft2s)
    alpha = np.tile(alpha[:,:,np.newaxis],(1,1,Nt))

    fout  = np.average(fdat,axis = 1,weights=alpha)
    out = unmask(fout,mask)
    print 'Out shape is ', out.shape
    return out

def wthresh(a, thresh):
    #Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

"""
# Wrap it in a function that gives me more context:
def ipsh():
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)

    frame = inspect.currentframe().f_back
    msg   = 'Stopped at {0.f_code.co_filename} at line {0.f_lineno}'.format(frame)

    # Go back one level!
    # This is needed because the call to ipshell is inside the function ipsh()
    ipshell(msg,stack_depth=2)
"""



#Default threshold of .03 is assumed to be for input in the range 0-1...
#original matlab had 8 out of 255, which is about .03 scaled to 0-1 range
def go_dec(X, thresh=.03, rank=2, power=1, tol=1e-3, max_iter=100, random_seed=0, verbose=True):
    print '++Starting Go Decomposition'
    m, n = X.shape
    if m < n:
        X = X.T
    m, n = X.shape
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)
    while True:
        Y2 = random_state.randn(n, rank)
        for i in range(power+1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1);
        Q, R = qr(Y2)
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = norm(T.ravel(), 2)
        if (err < tol) or (itr >= max_iter):
            break
        L += T
        itr += 1
    #Is this even useful in soft GoDec? May be a display issue...
    G = X - L - S
    if m < n:
        L = L.T
        S = S.T
        G = G.T
    if verbose:
        print "Finished at iteration %d" % (itr)
    return L, S, G

def dwtmat(mmix):
    #ipdb.set_trace()
    print "++Wavelet transforming data"
    lt = len(np.hstack(pywt.dwt(mmix[0],'db2')))
    mmix_wt = np.zeros([mmix.shape[0],lt])
    for ii in range(mmix_wt.shape[0]):
        wtx = pywt.dwt(mmix[ii],'db2')
        #print len(wtx[0]),len(wtx[1])
        cAlen = len(wtx[0])
        mmix_wt[ii] = np.hstack(wtx)
    return mmix_wt,cAlen

def idwtmat(mmix_wt,cAl):
    print "++Inverse wavelet transforming"
    lt = len(pywt.idwt(mmix_wt[0,:cAl],mmix_wt[0,cAl:],'db2',correct_size=True))
    mmix_iwt = np.zeros([mmix_wt.shape[0],lt])
    for ii in range(mmix_iwt.shape[0]):
        mmix_iwt[ii] = pywt.idwt(mmix_wt[ii,:cAl],mmix_wt[ii,cAl:],'db2',correct_size=True)
    return mmix_iwt

def wgo_dec(X, thresh=.03, rank=2, power=1, tol=1e-3, max_iter=100, random_seed=0, verbose=True):
    #mmix_gd = go_dec(mmix,thresh=mmix.std()*2.5,power=8,rank=2)
    #mmix_dn = mmix_gd[2]
    X_wt,cal = dwtmat(X)
    X_wgd = go_dec(X_wt,X_wt.std()*thresh,rank,power,tol,max_iter,random_seed,verbose)
    return idwtmat(X_wgd[0],cal),idwtmat(X_wgd[1],cal),idwtmat(X_wgd[2],cal)

def GreGoDec(D,ranks,tau,tol,inpower,k):

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                 Greedy Semi-Soft GoDec Algotithm (GreBsmo)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %INPUTS:
    %X: nxp data matrix with n samples and p features
    %rank: rank(L)<=rank
    %tau: soft thresholding
    %inpower: >=0, power scheme modification, increasing it lead to better
    %k: rank stepsize
    %accuracy and more time cost
    %OUTPUTS:
    %L:Low-rank part
    %S:Sparse part
    %RMSE: error
    %error: ||X-L-S||/||X||
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %REFERENCE:
    % Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix
    % Decomposition in Noisy Case", ICML 2011
    % Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and
    % Smoothing", AISTATS 2013.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Tianyi Zhou, 2013, All rights reserved.
    """


    def rank_estimator_adaptive(inv):
        if inv['estrank'] == 1:
            dR = abs(diag(R)); drops = dR[:-1]/dR[1:];
            print dR.shape
            dmx = max(drops);
            imx = argmax(drops);
            rel_drp = (inv['rankmax']-1)*dmx/(sum(drops)-dmx);
            #ipdb.set_trace()
            if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or inv['itr_rank'] > maxitr_reduce_rank: #%bar(drops); pause;
                inv['rrank'] = max([imx, floor(0.1*inv['rankmax']), rank_min]);
                #inv['error'][ii] = norm(res)/normz;
                inv['estrank'] = 0
		inv['itr_rank'] = 0;
            return inv

    #set rankmax and sampling dictionary
    rankmax=max(ranks)
    outdict={}
    rks2sam =[int(round(rk)) for rk in (np.array(ranks)/k)]
    rks2sam = sorted(rks2sam)

    #matrix size
    m,n=D.shape; # ok
    if m<n: D=D.T; # ok
    normD=norm(D[:]); # CHECK norm types

    #initialization of L and S by discoverying a low-rank sparse SVD and recombining
    rankk=int(round(rankmax/k)); # ok
    error=zeros(max(rankk*inpower,1)+1); # ok
    print error
    X,s,Y=svds(D,k,which='LM'); # CHECK svds
    s = diag(s);
    #embed()
    X=X.dot(s);   #CHECK dot notation
    L=X.dot(Y);    #CHECK dot notation
    S=wthresh(D-L,tau);  #ok
    T=D-L-S; # ok
    error[0]=norm(T[:])/normD; # CHECK norm types
    iii=1;
    stop=False;
    alf=0;
    estrank = -1


    #tic;
    #for r=1:rankk
    for r in range(1,rankk+1): # CHECK iterator range
        # parameters for alf
        rrank=rankmax;
        estrank = 1;
        rank_min =  1;
        rk_jump = 10;
        alf=0;increment=1;
        itr_rank = 0; minitr_reduce_rank = 5;    maxitr_reduce_rank = 50;
        if iii==inpower*(r-2)+1: iii=iii+inpower;
        for iteri in range(inpower+1):
            print 'r %i, iteri %i, rrank %i, alf %i' % (r,iteri,rrank,alf)
            #Update of X
            X=L.dot(Y.T);   #CHECK dot notation
            #if estrank==1:
            #    qro=qr(X,mode='economic');   #CHECK qr output formats    #stopping here on 1/12
	    #	X = qro[0];
	    #	R = qro[1];
            #else:
            X,R=qr(X,mode='reduced');  #CHECK qr output formats

            #Update of Y
            Y=X.T.dot(L); #CHECK dot notation
            L=X.dot(Y);   #CHECK dot notation

            #Update of S
            T=D-L; # ok
            S=wthresh(T,tau); # ok

            #Error, stopping criteria
            T=T-S;   # ok
            ii=iii+iteri-1; # ok
            #embed()
            error[ii]=norm(T[:])/normD;
            if error[ii]<tol:
                stop=True;
                break;

            # adjust estrank
            if estrank >= 1:
                outr = rank_estimator_adaptive(locals())
            	estrank,rankmax,itr_rank,error,rrank = [outr['estrank'],outr['rankmax'],outr['itr_rank'],outr['error'],outr['rrank']]

            if rrank != rankmax:
                rankmax = rrank
                if estrank ==0:
                    alf = 0;
                    continue;

            # adjust alf
            ratio=error[ii]/error[ii-1];
	    if np.isinf(ratio): ratio=0;
            print ii, error, ratio
            if ratio >= 1.1:
                increment = max(0.1*alf, 0.1*increment);
                X = X1; Y = Y1; L = L1; S = S1; T = T1; error[ii] = error[ii-1];
                alf = 0;
            elif ratio > 0.7:
                increment = max(increment, 0.25*alf);
                alf = alf + increment;

            # Update of L
            print 'updating L'
            X1=X;Y1=Y;L1=L;S1=S;T1=T;
            #ipdb.set_trace()
            L=L+((1+alf)*(T));

            # Add coreset
            if iteri>8:
                if mean(error[ii-7:ii+1])/error[ii-8]>0.92:
                    iii=ii;
                    sf=X.shape[1];
                    if Y.shape[0]-sf>=k:
                        Y=Y[:sf,:];
                    break;

        if r in rks2sam:
            L=X.dot(Y);
            if m<n :
                L=L.T;
                S=S.T;
            outdict[r*k] = [L,D-L,D-L-T]

        # Coreset
        if not stop and r<rankk:
            v=randn(k,m).dot(L);
            Y=vstack([Y,v]); #correct this

        # Stop
        if stop: break;

    error[error==0]=None;

    return outdict

def tedgodec(ste=0,ranks=[2],drank=2,inpower=2,thresh=10,max_iter=500,rmu_data=None,norm_mode=None):
    nx,ny,nz,ne,nt = catd.shape
    if ne==1: ste=1
    ste = np.array([int(ee) for ee in str(ste).split(',')])
    if len(ste) == 1 and ste[0]==-1:
        print "-Computing PCA of optimally combined multi-echo data"
        OCcatd = optcom(catd,t2s,tes,mask)
        OCmask = makemask(OCcatd[:,:,:,np.newaxis,:])
        d = fmask(OCcatd,OCmask)
        eim = eimask(d[:,np.newaxis,:])
        eim = eim[:,0]==1
        d = d[eim,:]
        #ipdb.set_trace()
    elif len(ste) == 1 and ste[0]==0:
        print "-Computing PCA of spatially concatenated multi-echo data"
        ste = np.arange(ne)
        d = np.float64(fmask(catd,mask))
        eim = eimask(d)==1
        d = d[eim]
    else:
        print "-Computing PCA of TE #%s" % ','.join([str(ee) for ee in ste])
        d = np.float64(np.concatenate([fmask(catd[:,:,:,ee,:],mask)[:,np.newaxis,:] for ee in ste-1],axis=1))
        eim = eimask(d)==1
        eim = np.squeeze(eim)
        d = np.squeeze(d[eim])

    #Make the unmask
    eimum = np.atleast_2d(eim)
    eimum = np.transpose(eimum,np.argsort(np.atleast_2d(eim).shape)[::-1])
    eimum = np.array(np.squeeze(unmask(eimum.prod(1),mask)),dtype=np.bool)

    if norm_mode == 'psc':
        #Convert to PSC
        rmu = rmu_data[eimum].mean(-1)
        dnorm = ((d/rmu[:,np.newaxis])-1)*100
        thresh = 0.1
    elif norm_mode == 'dm':
        #Demean
        rmu = d.mean(-1)
        dnorm = d - rmu[:,np.newaxis]
    elif norm_mode == 'vn':
        rmu = d.mean(-1)
        rstd = d.std(-1)
        dnorm = (d - rmu[:,np.newaxis])/rstd[:,np.newaxis]
    else:
        dnorm = d

    #GoDec
    out = {}
    if options.wavelet: out[ranks[0]] = list(wgo_dec(dnorm,rank=ranks[0],thresh=thresh))
    else: out[ranks[0]] = list(go_dec(dnorm,rank=ranks[0],thresh=thresh))

    #GreGoDec
    #out = GreGoDec(dnorm,ranks,1,1e-7,inpower,drank)

    if norm_mode == 'psc':
        for ii in range(len(out)): out[ii] = ((out[ii]/100)+1)*rmu[:,np.newaxis]
    elif norm_mode == 'dm':
        #Remean
        out[0] = out[0]+rmu[:,np.newaxis]
    elif norm_mode == 'vn':
        for rr in out.keys():
                out[rr][0] = out[rr][0]*rstd[:,np.newaxis]+rmu[:,np.newaxis]
                out[rr][1] = out[rr][1]*rstd[:,np.newaxis]
                out[rr][2] = out[rr][2]*rstd[:,np.newaxis]

    return out,eimum

def dogs(ranks,norm_mode=None,drank=2,inpower=2):
    gdoutm,eimum = tedgodec(ste=0,ranks=ranks,inpower=inpower,thresh=thresh,max_iter=500,norm_mode=norm_mode)
    for rank in sorted(gdoutm.keys()):
        gdout = gdoutm[rank]
        artout = unmask(gdout[0],eimum)
        sparseout = unmask(gdout[1],eimum)
        noiseout = unmask(gdout[2],eimum)

        if options.norm_mode == None:
            name_norm_mode = ''
        else:
            name_norm_mode = 'n%s' % options.norm_mode
        if options.wavelet: name_norm_mode = 'w%s' % name_norm_mode
        suffix='%sr%ik%ip%it%i' % (name_norm_mode,rank,drank,inpower,thresh)
        #niwrite(dnout,aff,'dn_%s.nii' % suffix)
        niwrite(artout,aff,'lowrank_%s.nii' % suffix)
        niwrite(sparseout,aff,'sparse_%s.nii' % suffix)
        niwrite(noiseout,aff,'noise_%s.nii' % suffix)

if __name__=='__main__':

    parser=OptionParser()
    parser.add_option('-d',"--orig_data",dest='data',help="Spatially Concatenated Multi-Echo Dataset",default=None)
    parser.add_option('-e',"--TEs",dest='tes',help="Echo times (in ms) ex: 15,39,63",default=None)
    parser.add_option('-r',"--rank",dest='rank',help="Rank of low rank component",default=2)
    parser.add_option('-k',"--increment",dest='drank',help="Rank search step size",default=2)
    parser.add_option('-p',"--power",dest='power',help="Power for power method",default=2)
    parser.add_option('-w',"--wavelet",dest='wavelet',help="Wavelet transform before GoDec",default=False,action='store_true')
    parser.add_option('-t',"--thresh",dest='thresh',help="Poewr for power method",default=2)
    parser.add_option('-n',"--norm_mode",dest='norm_mode',help="Normalization mode",default='vn')
    parser.add_option('',"--label",dest='label',help="Label for output directory.",default=None)

    (options,args) = parser.parse_args()

    print "-- GoDec fMRI Denoising %s--" % __version__

    if options.data==None:
        print "*+ Need dataset name, use -h for help."
        sys.exit()

    if options.tes==None:
        tes = [30.]
        ne = 1
    else:
        tes = np.fromstring(options.tes,sep=',',dtype=np.float32)
        ne = tes.shape[0]

    if options.wavelet: import pywt

    print "++ Loading Data"
    catim  = nib.load(options.data)
    head   = catim.get_header()
    head.extensions = []
    head.set_sform(head.get_sform(),code=1)
    aff = catim.get_affine()
    catd = cat2echos(catim.get_data(),ne)
    nx,ny,nz,Ne,nt = catd.shape

    mu  = catd.mean(axis=-1)
    sig  = catd.std(axis=-1)
    mask = makemask(catd)

    """Parse options, prepare output directory"""
    if options.label!=None: dirname='%s' % '.'.join(['GD',options.label])
    else: dirname='GD'
    os.system('mkdir %s' % dirname)

    #embed()

    #Set threshold
    if not options.thresh:
        thresh=np.median(mu[mu!=0])*0.01
    else:
        thresh=float(options.thresh)
    ranks=[int(rr) for rr in options.rank.split(',')]
    drank=int(options.drank)
    inpower=int(options.power)
    dogs(ranks,drank=drank,inpower=inpower,norm_mode=options.norm_mode)

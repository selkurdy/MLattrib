'''
mlattrib.py ver1.01



    Usage: python swattrib.py sattrib | featurescale | wscalecols | dataprep | listcsvcols
        drop | wattrib | wamerge | seiswellattrib  | scatterplot | qclin  |  linreg  | testCmodels
         | logisticreg | GaussianNaiveBayes | clustertest | clustering |  tSNE




python mlattrib.py  CatBoostRegressor BAH010_46smth.csv BAH010_46smth.csv --modelcolsrange 1 21 --predictioncolsrange 1 21 --samplemodel 0.5 --modeltargetcol 21 --predictionidcol 0

python mlattrib.py  KNNfitpredict BAH010_46smth.csv BAH010_46smth.csv --modelcolsrange 1 21 --predictioncolsrange 1 21 --samplemodel 0.5 --modeltargetcol 21 --predictionidcol 0

testcsv = prepcsv('BAH010_46_coded.csv',colsrange=[1,10],targetcol=22,targetencode=True)
X,cn = testcsv.extract_data()
y,yh = testcsv.extract_target()

python mlattrib.py  logisticreg BAH010_46_coded.csv BAH010_46_coded.csv --modelcolsrange 1 21 --predictioncolsrange 1 21 --samplemodel 0.5 --modeltargetcol 22 --coded --predictionidcol 0


modelcsv = prepcsv('BAH010_46_coded.csv',idcol =0,targetcol=-1,
    colsrange=[1,10],colsselect = None,
    scalefeatures = True,scaletype= 'standard',
    qcut=False,targetencode=False,coded=True,
    scalesave=True,sample=0.5)


modelcsv = prepcsv('BAH010_46_coded.csv',idcol =0,targetcol=-1,
colsrange=[1,10],colsselect = None,
scalefeatures = True,scaletype= 'standard',
qcut=False,nqcutclasses = 4, targetencode=True,coded=False,
scalesave=True,sample=0.5)

python mlattrib.py GaussianNaiveBayes wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --predictioncolsrange 3 15

python mlattrib.py logisticreg wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --predictioncolsrange 3 15

python mlattrib.py CatBoostClassifier wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --predictioncolsrange 3 15

python mlattrib.py  linfitpredict BAH010_46smth.csv BAH010_46smth.csv --modelcolsrange 1 21 --predictioncolsrange 1 21 --samplemodel 0.5 --modeltargetcol 21 --predictionidcol 0

python mlattrib.py PCAanalysis wattrib.csv  --modelcolsrange 4 16

python mlattrib.py PCAanalysis sattrib.csv  --modelcolsrange 3 15

python mlattrib.py PCAfilter sattrib.csv  --modelcolsrange 3 15 --ncomponents 6 --modelidcol 2

python mlattrib.py linreg wattrib.csv  --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17

python mlattrib.py KNNtest wattrib.csv  --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17

python mlattrib.py linfitpredict wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17 --predictioncolsrange 3 15 --predictionidcol 2

python mlattrib.py KNNfitpredict wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17 --predictioncolsrange 3 15 --predictionidcol 2

python mlattrib.py qclin wattrib.csv  --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17

python mlattrib.py qclin wattrib.csv  --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17 --heatonly

python mlattrib.py featureranking wattrib.csv  --modelcolsrange 4 16 --modelidcol 0 --modeltargetcol 17 --testfeatures mutualinforeg

python mlattrib.py CatBoostRegressor wattrib.csv sattrib.csv --modelcolsrange 4 16 --modelidcol 0 --predictioncolsrange 3 15


'''

import sys, os.path
import argparse
import shlex
import time
import datetime,time
import warnings
import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sts
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from scipy.spatial import distance
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.path as mplpath
import seaborn as sns
from collections import Counter


from pandas.tools.plotting import scatter_matrix  #->deprecated
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn import preprocessing
#from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error,log_loss
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import r2_score                 #Coefficient of Determination
from sklearn.cluster import KMeans,DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
# from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.semi_supervised import LabelSpreading
from sklearn.externals import joblib
# from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
try:
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
except ImportError:
    print('***Warning:CatBoost is not installed')

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor



def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


def plot_classifier(classifier, X, y,xcol0=0,xcol1=1):
    # define ranges to plot the figure
    x_min, x_max = min(X[:, xcol0]) - 1.0, max(X[:, xcol0]) + 1.0
    y_min, y_max = min(X[:, xcol1]) - 1.0, max(X[:, xcol1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()





def qhull(sample):
    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
    	axis = sample[:,0]
    	base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
    	return link(dome(sample, base),dome(sample, base[::-1]))
    else:
        return sample


    """
    Flat file data with xy and variable (vr)
    Input polygon to list data filtered only inside polygon
    Input horizon flat file with xy only to back interpolate on data set
    Jan 23 2012

    """


def pip(x,y,poly):
   # check if point is a vertex
   """
   if (x,y) in poly:
        return True
    """
   # check if point is on a boundary
   for i in range(len(poly)):
      p1 = None
      p2 = None
      if i==0:
         p1 = poly[0]
         p2 = poly[1]
      else:
         p1 = poly[i-1]
         p2 = poly[i]
      if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
         return True

   n = len(poly)
   inside = False

   p1x,p1y = poly[0]
   for i in range(n+1):
      p2x,p2y = poly[i % n]
      if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
            if x <= max(p1x,p2x):
               if p1y != p2y:
                  xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
               if p1x == p2x or x <= xints:
                  inside = not inside
      p1x,p1y = p2x,p2y

   if inside: return True
   else: return False


#...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


def idw(xy,vr,xyi):

    N = vr.size
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 2  # weights ~ 1 / distance**p
    invdisttree = Invdisttree( xy, vr, leafsize=leafsize, stat=1 )
    interpol = invdisttree( xyi, nnear=Nnear, eps=eps, p=p )
    return interpol


def gridlistin(fname,xyvcols=[0,1,2],nheader=0): #used for single coef per file
    xyv=np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    #filter surfer null values by taking all less than 10000, arbitrary!!
    xyv = xyv[xyv[:,2]<10000.0]
    #xya = xya[~xya[:,2]==  missing]
    return xyv[:,0],xyv[:,1],xyv[:,2]




def map2ddata(xy,vr,xyi,radius,maptype):
    stats=sts.describe(vr)
    statsstd=sts.tstd(vr)
    if maptype == 'idw':
        vri=idw(xy,vr,xyi)
    elif maptype =='nearest':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='nearest')
    elif maptype == 'linear':
        #                vri=griddata(xy,vr,(xyifhull[:,0],xyifhull[:,1]),method='linear')
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='linear')
    elif maptype == 'cubic':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='cubic')
    elif maptype =='rbf':
        rbf=Rbf(xy[:,0],xy[:,1],vr)
        vri= rbf(xyi[:,0],xyi[:,1])
    elif maptype =='avgmap':
        vri=dataavgmap(xy,vr,xyi,cmdl.radius)
    elif maptype =='triang':
        linearnd=LinearNDInterpolator(xy,vr,stats[2])
        vri= linearnd(xyi)
    elif maptype == 'ct':
        ct=CloughTocher2DInterpolator(xy,vr,stats[2])
        vri=ct(xyi)
    return vri

def filterhullpolygon(x,y,polygon):
    xf=[]
    yf=[]
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
    return np.array(xf),np.array(yf)


def filterhullpolygon_wid(x,y,id0,id1,polygon):
    xf=[]
    yf=[]
    id0f=[] #for well name
    id1f=[] #for z values
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
            id0f.append(id0[i]) #well name
            id1f.append(id1[i]) #well attribute or porosity
    return np.array(xf),np.array(yf),id0f,np.array(id1f)



def filterhullpolygon_mask(x,y,polygon):
    ma=[]
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            ma.append(True)
        else:
            ma.append(False)
    return np.array(ma)




def qcattributes(dfname,pdffname=None,deg=1,dp=3,scattermatrix=False,cmdlsample=None):
    '''
    General function to establish linear fit relationships between each predictor and singular target
    The second variable changes it from 1=linear to >1=Polynomial Fit

    '''
    with PdfPages(pdffname) as pdf:
        for i in range((dfname.shape[1])-1):
            xv =dfname.iloc[:,i].values
            yv =dfname.iloc[:,-1].values
            xtitle = dfname.columns[i]
            ytitle = dfname.columns[-1]
            xrngmin,xrngmax = xv.min(),xv.max()
            #print(xrngmin,xrngmax)
            xvi = np.linspace(xrngmin,xrngmax)
            #print(xrng)
            qc = np.polyfit(xv,yv,deg)
            if deg == 1:
                print('Slope: %5.3f, Intercept: %5.3f'%(qc[0],qc[1]))
            else:
                print(qc)
            yvi = np.polyval(qc,xvi)
            p0 = plt.scatter(xv,yv,alpha=0.5)
            p0 = plt.plot(xvi,yvi,c='red')
            p0 = plt.xlabel(xtitle)
            p0 = plt.ylabel(ytitle)
            #commenting out annotation : only shows on last plot!!
            if deg == 1:
                plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),\
                            xy =(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

            #plt.show()
            pdf.savefig()
            plt.close()

        if scattermatrix:
            dfnamex = dfname.sample(frac=cmdlsample).copy()
            scatter_matrix(dfnamex)
            pdf.savefig()
            plt.show()
            plt.close()



def savefiles(seisf = None,sdf = None,sxydf =None,
                wellf = None, wdf = None, wxydf  = None,
                outdir=None,ssuffix='',wsuffix='',name2merge=None):
    if seisf:
        dirsplit,fextsplit= os.path.split(seisf)
        fname1,fextn= os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2= os.path.split(name2merge)
            fname2,fextn2= os.path.splitext(fextsplit2)
            fname = fname1 +'_'+ fname2
        else:
            fname =fname1

        if outdir:
            slgrf = os.path.join(outdir,fname) + ssuffix +".csv"
        else:
            slgrf = os.path.join(dirsplit,fname) + ssuffix + ".csv"
        #if not sdf.empty:
        if  isinstance(sdf,pd.DataFrame) :
            sdf.to_csv(slgrf,index = False)
            print('Successfully generated %s file'  % slgrf)

        if outdir:
            slgrftxt = os.path.join(outdir,fname) + ssuffix + ".txt"
        else:
             slgrftxt = os.path.join(dirsplit,fname) + ssuffix + ".txt"
        if  isinstance(sxydf,pd.DataFrame) :
            sxydf.to_csv(slgrftxt,sep=' ',index=False)
            print('Successfully generated %s file'  % slgrftxt)


    if wellf:
        dirsplit,fextsplit= os.path.split(wellf)
        fname1,fextn= os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2= os.path.split(name2merge)
            fname2,fextn2= os.path.splitext(fextsplit2)
            fname = fname1 +  '_' +fname2
        else:
            fname =fname1


        if outdir:
            wlgrf = os.path.join(outdir,fname) + wsuffix + ".csv"
        else:
            wlgrf = os.path.join(dirsplit,fname) + wsuffix + ".csv"
        if isinstance(wdf,pd.DataFrame):
            wdf.to_csv(wlgrf,index = False)
            print('Successfully generated %s file'  % wlgrf)

        if outdir:
            wlgrftxt = os.path.join(outdir,fname) + wsuffix + ".txt"
        else:
            wlgrftxt = os.path.join(dirsplit,fname) + wsuffix + ".txt"
        if isinstance(wxydf,pd.DataFrame):
            wxydf.to_csv(wlgrftxt,sep=' ',index=False)
            print('Successfully generated %s file'  % wlgrftxt)





def listfiles(flst):
    for fl in flst:
        print(fl)


# calculate the fpr and tpr for all thresholds of the classification
def plot_roc_curve(y_test, preds,poslbl,hideplot=False,pdfsave=None):
    fpr, tpr, threshold = roc_curve(y_test, preds,pos_label=poslbl)
    roc_auc = auc(fpr, tpr)
    with PdfPages(pdfsave) as pdf:
        plt.figure(figsize=(8,8))
        plt.title('ROC Curve  %1d'% poslbl)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        if not hideplot:
            plt.show()
        plt.close()



#**********************************
#****Feb 26, 2018
#final version that works with regression
#and works with classification
class prepcsv():
    """
    work on one csv file to scale all features
    If there is an id column e.g. Z or a target column then
    it (or they) are removed first, then features are scaled then added back
    If target column to be used for classification then targetcode is set to True
    I target used for regression we can scale it also but seperately



    """
    def __init__(self,csvfname,idcols=None,targetcol=None,colsrange=None,colsselect=None,
                scalefeatures= True,scaletype='standard',scalesave=True,targetencode=False,
                scaletarget= False,targetscaletype='standard',
                loadencoder=False,qcut=None,nqcutclasses= 3,coded=False,sample=None):
        """
        scalesave is true if you scale data and dump scaler to be used later on
        test or validate data
        scalesave is false if you have test/validate data
        Assumption is scaler coefs are dumped to working dir and will be accessed again
        from same dir
        coded flag to say target is already coded. default = False, i.e. for
        classification use qcut

        xyzout generates a dataframe of idcols and target


        #extract and scale columns of data only
        #dfin,dfs = modelcsv.extract_df_workflow()
        #X,colnames = modelcsv.extract_data()
        X,colnames,dfin= modelcsv.extract_scale_cols()
        y,ycolname = modelcsv.extract_target()
        dfidadded = modelcsv.addback_idcols(dfin)

        #addback_targetcol adds to the scaled features
        dfidtargetadded,tcolnum = modelcsv.addback_targetcol(dfidadded)

        #idtarget_merge designed to merge well x y z with original target
        # and predicted target to create a new dataframe then save it as a txt
        #file for Petrel import

        #xyzdf = modelcsv.idtarget_merge(predicteddf=dfin,predicteddfcols=[3,1,6])
        #xyzdf = modelcsv.idtarget_merge(predicteddf=dfin,predicteddfcols=3)
        xyzdf = modelcsv.idtarget_merge()

        #no need for scale_data because it is included in extract_data
        #Xs,colnamess =modelcsv.scale_data()

        """
        self.csvfname = csvfname
        self.idcols = idcols
        self.targetcol = targetcol
        self.colsrange =colsrange
        self.colsselect =colsselect
        self.scalefeatures =scalefeatures
        self.scaletype = scaletype
        self.scalesave = scalesave
        self.targetencode = targetencode
        self.scaletarget =scaletarget
        self.targetscaletype= targetscaletype
        self.loadencoder =loadencoder
        self.qcut =qcut
        self.nqcutclasses =nqcutclasses
        self.coded = coded
        self.le_filename = "lblencoder.save"
        self.sample = sample

        self.allattrib = pd.read_csv(self.csvfname)
        self.check_sample()

        self.csvhead()
        self.dropnulls()

    def check_sample(self):
        if self.sample:
            self.allattrib = self.allattrib.sample(frac=self.sample).copy()
            return self.allattrib
        else:
            return self.allattrib

    def csvhead(self):
        #list head of csv
        print(self.allattrib.shape)
        #print(self.allattrib.head())

    def dropnulls(self):
        if self.allattrib.isnull().values.any():
            print('Warning: Null Values in the file will be dropped')
            self.allattrib.dropna(inplace=True)
        else:
            print('No Nulls in csv file {}'.format(self.csvfname))
        return self.allattrib


    def list_csvcols(self):
        #list all columns in csv
        print(list(enumerate(self.allattrib.columns)))

    def idcols_extract(self,allattrib):
        #extract id column data and header
        #extract to a dataframe
        self.iddf = allattrib[allattrib.columns[self.idcols]].copy()
        # print('iddf shape:',self.iddf.shape)
        return self.iddf

    def targetcol_name(self):
        #extract target column data and header
        self.target_name = self.allattrib.columns[self.targetcol]
        return self.target_name

    def targetcol_data(self):
        #extract target column data and header
        self.target_data = self.allattrib[self.allattrib.columns[self.targetcol]].values
        return self.target_data

    def target_encode(self):
        "Target used for classification"
        if self.loadencoder:
            self.le = joblib.load(self.le_filename)
        else:
            self.le = LabelEncoder()
            self.le.fit(self.target_data)
            self.classes = list(self.le.classes_)
            joblib.dump(self.le, self.le_filename)
        self.target_code =self.le.transform(self.target_data)
        #self.dfcoded = pd.DataFrame({self.target_name : self.target_code},dtype='category')
        #self.allattrib['Target_coded'] = self.target_code
        print('Classes:',self.classes)
        #return only the coded df
        return self.target_code,self.classes

    def target_qcut(self):
        """
        Use for numertic columns to codes using qcut
        """
        # self.probacolnames = ['Class%d'%i for i in range(self.nqcutclasses)]
        self.classes = ['Class%d'%i for i in range(self.nqcutclasses)]

        #codes,qbins = pd.qcut(self.allattrib[self.allattrib.columns[self.targetcol]],
                                        #self.qcut,labels=self.probacolnames,retbins=True)
        codes,qbins = pd.qcut(self.target_data,self.nqcutclasses,labels=self.classes,retbins=True)
        self.target_name = self.target_name + '_qcoded'
        dfcodes = pd.DataFrame({self.target_name : codes},dtype='category')
        self.target_code = dfcodes[self.target_name].cat.codes
        print('Bins:',qbins)

        return self.target_code,self.classes


    def target_coded(self):
        self.classes = self.allattrib[self.allattrib.columns[self.targetcol]].unique()
        self.target_code = self.allattrib[self.allattrib.columns[self.targetcol]]
        print('In target_coded:',self.target_data.shape,self.target_code.shape)
        return self.target_code,self.classes


    def target_scale(self):
        "Target used for regression"
        if self.targetscaletype =='standard':
            self.target_scaled = StandardScaler().fit_transform(self.target_data)
        elif self.targetscaletype== 'quniform':
            self.target_scaled = QuantileTransformer(output_distribution='uniform').fit_transform(self.target_data)
        else:
            self.target_scaled = QuantileTransformer(output_distribution='normal').fit_transform(self.target_data)
        return self.target_scaled

    def extract_scale_cols(self):
        """
        #given either a list of columns or a range of columns
        #returns all data cols numpy array and dataframe
        """
        if self.colsrange:
            #extract a range of columns and return numpy array
            print('Well Predictors From col# %d to col %d' %(self.colsrange[0],self.colsrange[1]))
            self.X = self.allattrib[self.allattrib.columns[self.colsrange[0]: self.colsrange[1]+1]].values
            self.colnames = self.allattrib[self.allattrib.columns[self.colsrange[0]: self.colsrange[1]+1]].columns
        else:
            #extract a list of columns and return a numpy array
            print('Well Predictor cols',self.colsselect)
            self.X = self.allattrib[self.allattrib.columns[self.colsselect]].values
            self.colnames = self.allattrib[self.allattrib.columns[self.colsselect]].columns
        if self.scalefeatures:
            scaler_filename = "scaler.save"
            if self.scalesave:
                if self.scaletype =='standard':
                    scaler = StandardScaler().fit(self.X)
                elif self.scaletype== 'quniform':
                    scaler = QuantileTransformer(output_distribution='uniform').fit(self.X)
                else:
                    scaler = QuantileTransformer(output_distribution='normal').fit(self.X)
                joblib.dump(scaler, scaler_filename)
            else:
                scaler = joblib.load(scaler_filename)
            self.X = scaler.transform(self.X)
            self.df_datascaled = pd.DataFrame(self.X,columns=self.colnames)

        else:
            self.df_datascaled = pd.DataFrame(self.X,columns=self.colnames)
        return self.X,self.colnames,self.df_datascaled

    def addback_idcols(self,datascaled_df):
        #dfid = self.allattrib[self.allattrib.columns[self.idcol]]
        self.idcols_extract(self.allattrib)
        self.df_idmerged = pd.concat([self.iddf,datascaled_df],axis=1,join='inner')
        return self.df_idmerged


    def addback_targetcol(self,datascaled_df):
        self.target_name = self.targetcol_name()
        self.target_data = self.targetcol_data()
        if self.scaletarget:
            dfscaled2 = pd.DataFrame({self.target_name : self.target_scaled})
        elif self.coded:
            dfscaled2 = pd.DataFrame({self.target_name : self.target_data},dtype='category')
        elif self.qcut:
            self.target_qcut()
            dfscaled2 = pd.DataFrame({self.target_name : self.target_code},dtype='category')
        elif self.targetencode:
            self.target_encode()
            self.target_name = self.target_name + '_encoded'
            dfscaled2 = pd.DataFrame({self.target_name : self.target_code},dtype='category')
        else:
            dfscaled2 = pd.DataFrame({self.target_name : self.target_data})

        self.df_targetmerged = pd.concat([datascaled_df,dfscaled2],axis=1,join='inner')
        return self.df_targetmerged,self.df_targetmerged.shape[1]


    def idtarget_merge(self,predicteddf=None,predicteddfcols=-1):
        """
        To be used only on prediction csv after adding predicted column(s)

        """
        self.idcols_extract(self.allattrib)
        if isinstance(predicteddf,pd.DataFrame):
            print(predicteddf.head())
            preddf = predicteddf[predicteddf.columns[predicteddfcols]].copy()
            self.df_idtarget_merged = pd.concat([self.iddf,preddf],axis=1,join='inner')
        else:
            self.df_idtarget_merged = pd.concat([self.iddf,self.df_datascaled],axis=1,join='inner')

        return self.df_idtarget_merged


    def extract_target(self):
        """
        extract target if requested
        return 4 entities:
        features, features column names
        target, target column name
        """
        self.target_name = self.targetcol_name()
        self.target_data = self.targetcol_data()
        if self.coded:
            self.target_coded()
            return self.target_data,self.classes

        elif self.targetencode:
            self.target_encode()
            return self.target_code,self.classes

        elif self.qcut:
            self.target_qcut()
            return self.target_code,self.classes

        elif self.scaletarget:
            return self.target_scaled,self.target_name

        else:
            return self.target_data,self.target_name





#**********************************



def process_dropcols(csvfile,
                    cmdlcols2drop=None,
                    cmdloutdir=None):


    allattrib = pd.read_csv(csvfile)
    #print(allattrib.head(5))
    #cols = allattrib.columns.tolist()
    allattrib.drop(allattrib.columns[[cmdlcols2drop]],axis=1,inplace=True)
    #print(allattrib.head(5))
    cols = allattrib.columns.tolist()

    savefiles(seisf = csvfile,
                sdf = allattrib,
                outdir = cmdloutdir,
                ssuffix = '_drpc')



def process_listcsvcols(csvfile):
    data =pd.read_csv(csvfile)
    clist = list(enumerate(data.columns))
    print(clist)


def process_PCAanalysis(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdloutdir=None,
            cmdlhideplot = False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()

    pca = PCA()
    pca.fit(X)
    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfsave = os.path.join(cmdl.outdir,fname) +"_pca.pdf"
    else:
        pdfsave = os.path.join(dirsplit,fname) +"_pca.pdf"

    # Plot the explained variances
    features = range(pca.n_components_)
    with PdfPages(pdfsave) as pdf:
        plt.figure(figsize=(8,8))
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.title('Elbow Plot')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()



def process_PCAfilter(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlkind='standard',
            cmdlscalesave=True,
            cmdlncomponents= None,
            cmdloutdir=None,
            cmdlhideplot = False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data


    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    if cmdlmodeltargetcol:
        y,ycolname = modelcsv.extract_target()

    # Create a PCA instance: pca
    if not cmdlncomponents:
        pca = PCA(X.shape[1] )
        colnames = list()
        #[colnames.append('PCA%d'%i) for i in range(X.shape[1] -1)]
        [colnames.append('PCA%d'%i) for i in range(X.shape[1])]
    else:
        pca = PCA(cmdlncomponents )
        colnames = list()
        #[colnames.append('PCA%d'%i) for i in range(cmdl.ncomponents -1)]
        [colnames.append('PCA%d'%i) for i in range(cmdlncomponents)]

    CX = pca.fit_transform(X)
    print('cx shape',CX.shape,'ncolumns ',len(colnames))

    cxdf = pd.DataFrame(CX,columns=colnames)


    if cmdlmodelidcol >= 0:
        cxdf = modelcsv.addback_idcols(cxdf)
    if cmdlmodeltargetcol:
        cxdf,tcolnum = modelcsv.addback_targetcol(cxdf)



    savefiles(seisf = modeldatacsv,
                sdf = cxdf,
                outdir = cmdloutdir,
                ssuffix = '_pca')


def process_scattermatrix(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlmodelscalesave=True,
            cmdlkind='standard',
            cmdloutdir=None,
            cmdlhideplot = False):

    #scalesave has to be True to save scaler to be used later by predictiondata
    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=cmdlmodelscalesave,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)


    # g = sns.PairGrid(dfscaled)
    # g = g.map_diag(plt.hist)
    # g = g.map_offdiag(plt.scatter)
    # plt.show()
    sctrm = scatter_matrix(dfin, alpha=0.2, figsize=(6, 6), diagonal='kde')
    """
    fig = sctrm[0].get_figure()
    #****unable to get scatter matrix plot to save
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_sctrm.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_sctrm.pdf"
    fig.savefig(pdfcl)
    """

def process_qclin(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpolydeg = 1,
            cmdlheatonly = False, #plot scattermatrix default=False
            cmdloutdir=None,
            cmdlhideplot = False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data


    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    """
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    """
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    plt.style.use('seaborn-whitegrid')
    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    dp = 3 #decimal places for display
    ytitle = dfin.columns[-1]
    if not cmdlheatonly:
        for i in range((dfin.shape[1])-1):

            xtitle = dfin.columns[i]
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_xp%s.pdf" %xtitle
            else:
                pdfcl = os.path.join(dirsplit,fname) +"_xp%s.pdf" %xtitle

            xv =dfin.iloc[:,i].values
            yv =dfin.iloc[:,-1].values
            xrngmin,xrngmax = xv.min(),xv.max()
            #print(xrngmin,xrngmax)
            xvi = np.linspace(xrngmin,xrngmax)
            #print(xrng)
            qc = np.polyfit(xv,yv,cmdlpolydeg)
            if cmdlpolydeg == 1:
                print('%s  vs %s  Slope: %5.3f, Intercept: %5.3f'%(xtitle,ytitle,qc[0],qc[1]))
            else:
                print('%s  vs %s ' %(xtitle,ytitle),qc)
            yvi = np.polyval(qc,xvi)
            p0 = plt.scatter(xv,yv,alpha=0.5)
            p0 = plt.plot(xvi,yvi,c='red')
            p0 = plt.xlabel(xtitle)
            p0 = plt.ylabel(ytitle)
            #commenting out annotation : only shows on last plot!!
            if cmdlpolydeg == 1:
                plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),\
                            xy =(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

            if not cmdlhideplot:
                plt.show()
            fig = p0.get_figure()
            fig.savefig(pdfcl)

    if cmdloutdir:
        pdfheat = os.path.join(cmdloutdir,fname) +"_heat.pdf"
    else:
        pdfheat = os.path.join(dirsplit,fname) +"_heat.pdf"

    plt.figure(figsize=(12,12))
    mask = np.zeros_like(dfin.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ht = sns.heatmap(dfin.corr(),
                vmin=-1, vmax=1,
                square=True,
                cmap='RdBu_r',
                mask=mask,
                linewidths=.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    if not cmdlhideplot:
        plt.show()
    fig = ht.get_figure()
    fig.savefig(pdfheat)








def process_linreg(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlscaletarget=None,   #scaling to predicted target
            cmdlscaleminmaxvalues =None,
            cmdloutdir=None,
            cmdlhideplot = False):



    # donotscale default is false from command line
    #scalesave has to be True to save scaler to be used later by predictiondata
    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)



    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    lm = LinearRegression()
    lm.fit(X, y)  # Fitting all predictors 'X' to the target 'y' using linear fit model

    # Print intercept and coefficients
    print ('Intercept: ',lm.intercept_)
    print ('Coefficients: ',lm.coef_)
    print ('R2 Score:',lm.score(X, y))


    # Calculating coefficients
    cflst = lm.coef_.tolist()
    #cflst.append(lm.intercept_)
    cflst.insert(0,lm.intercept_)
    cnameslst = colnames.tolist()
    #cnameslst.append('Intercept')
    cnameslst.insert(0,'Intercept')
    coeff = pd.DataFrame(cnameslst,columns=['Attribute'])
    coeff['Coefficient Estimate'] = pd.Series(cflst)


    pred = lm.predict(X)

    if cmdlscaletarget:
        if cmdlscaleminmaxvalues:
            ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
            print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
        else:
            ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        #mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        dfin['LRPred'] = predscaled
    else:
        dfin['LRPred'] = pred





    # Calculating Mean Squared Error
    mse = np.mean((pred - y)**2)
    print('MSE: ',mse)
    print('R2 Score:',lm.score(X,y))
    dfin['Predict'] = pred
    dfin['Prederr'] = pred - y

    ax =sns.regplot(x=y,y=pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Linear Regressor %s' %ycolname)
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_lreg.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_lreg.pdf"
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    savefiles(seisf = modeldatacsv,
                sdf = dfin,
                wellf = modeldatacsv,
                wdf = coeff,
                outdir = cmdloutdir,
                ssuffix ='_lr',
                wsuffix = '_lrcf')

def process_linfitpredict(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdlscaletarget = None,
            cmdlminmaxscale=None,
            cmdlscaleminmaxvalues =None,
            cmdloutdir=None,
            cmdlhideplot = False):


    #scalesave has to be True to save scaler to be used later by predictiondata
    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)
    #plt.style.use('seaborn-whitegrid')

    lm = LinearRegression()
    lm.fit(X, y)  # Fitting all predictors 'X' to the target 'y' using linear fit model
    ypred = lm.predict(X)
    # Print intercept and coefficients
    print ('Intercept: ',lm.intercept_)
    print ('Coefficients: ',lm.coef_)
    print ('R2 Score:',lm.score(X, y))

    # Calculating coefficients
    cflst = lm.coef_.tolist()
    #cflst.append(lm.intercept_)
    cflst.insert(0,lm.intercept_)
    cnameslst = colnames.tolist()
    #cnameslst.append('Intercept')
    cnameslst.insert(0,'Intercept')
    coeff = pd.DataFrame(cnameslst,columns=['Attribute'])
    coeff['Coefficient Estimate'] = pd.Series(cflst)

    pred = lm.predict(Xpred)
    if cmdlscaletarget:
        if cmdlscaleminmaxvalues:
            ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
            print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
        else:
            ymin,ymax = y.min(), y.max()

        mmscale = MinMaxScaler((ymin,ymax))
        #mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        dfinpred['LRPred'] = predscaled
    else:
        dfinpred['LRPred'] = pred
    # addback id column
    if cmdlmodelidcol >= 0:
        dfspred = predictioncsv.addback_idcols(dfinpred)


    #ax =plt.scatter(y,ypred)
    ax =sns.regplot(x=y,y=ypred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Linear Regressor %s' %ycolname)
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_lreg.pdf"
        xyplt = os.path.join(cmdloutdir,fname) +"_lregxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_lreg.pdf"
        xyplt = os.path.join(dirsplit,fname) +"_lregxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)
    xpltcols =['Actual','Predicted']
    # xpltdf = dfin[dfin.columns[cmdlmodelidcol]].copy()  #copy back model id col
    xpltdf = dfin[dfin.columns[0]].copy()
    #copy back model id col assuming idcol was added back at column 0
    xpltdf['Actual'] = y
    xpltdf['Predicted'] = ypred
    xpltdf.to_csv(xyplt,index=False)
    print('Sucessfully generated xplot file %s'  % xyplt)



    if cmdlpredictionidcol:
        dfinpred = predictioncsv.addback_idcols(dfinpred)
        idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)


    savefiles(seisf = predictiondatacsv,
                sdf = dfinpred,
                wellf = modeldatacsv,
                wdf = coeff,
                sxydf = idtargetdf,
                outdir = cmdloutdir,
                ssuffix ='_lr',
                wsuffix = '_lrcf',name2merge =modeldatacsv)



def process_KNNtest(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlkind='standard',
            cmdlcv=None,
            cmdloutdir=None,
            cmdlhideplot = False):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data


    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)

    k_values =np.array([n for n in range(1,21)])
    #print('kvalues:',k_values)
    mselist = []
    stdlist =[]
    for k in k_values:
        num_folds = cmdlcv
        kfold = KFold(n_splits=10, random_state=7)
        KNNmodel = KNeighborsRegressor(n_neighbors=k)
        scoring = 'neg_mean_squared_error'
        results = cross_val_score(KNNmodel, X, y, cv=kfold, scoring=scoring)
        print("K value: %2d  MSE: %.3f (%.3f)" % (k,results.mean(), results.std()))
        mselist.append(results.mean())
        stdlist.append(results.std())

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_knn.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_knn.pdf"

    ax = plt.plot(k_values,mselist)
    plt.xlabel('# of clusters')
    plt.ylabel('Neg Mean Sqr Error')
    plt.savefig(pdfcl)
    if not cmdlhideplot:
        plt.show()





def process_KNNfitpredict(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdlminmaxscale=None,
            cmdlscaleminmaxvalues =None,
            cmdloutdir=None,
            cmdlhideplot = False,
            cmdlkneighbors=10):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data


    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)

    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)
    #plt.style.use('seaborn-whitegrid')

    KNNmodel = KNeighborsRegressor(n_neighbors=cmdlkneighbors)
    KNNmodel.fit(X, y)

    ypred = KNNmodel.predict(X)

    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' %(mse))
    print('R2 Score: %.4f' %(KNNmodel.score(X,y)))

    pred = KNNmodel.predict(Xpred)
    if cmdlminmaxscale:
        if cmdlscaleminmaxvalues:
            ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
            print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
        else:
            ymin,ymax = y.min(), y.max()

        mmscale = MinMaxScaler((ymin,ymax))
        #mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        dfinpred['KNNPred'] = predscaled
    else:
        dfinpred['KNNPred'] = pred
    # addback id column
    if cmdlpredictionidcol >= 0:
        dfinpred = predictioncsv.addback_idcols(dfinpred)
    #ax =plt.scatter(y,ypred)
    sns.set(color_codes=True)
    ax =sns.regplot(x=y,y=ypred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    # plt.title('KNN Regressor %s' %dfin.columns[cmdlmodeltargetcol])
    plt.title('KNN Regressor %s' %ycolname)
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_knnreg.pdf"
        xyplt = os.path.join(cmdloutdir,fname) +"_knnregxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_knnreg.pdf"
        xyplt = os.path.join(dirsplit,fname) +"_knnregxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    xpltcols =['Actual','Predicted']
    # xpltdf = dfs.iloc[:,:3].copy()  #copy well x y
    # xpltdf = dfin[dfin.columns[cmdlmodelidcol]].copy()  #copy back model id col
    xpltdf = dfin[dfin.columns[0]].copy()  #copy back model id col
    xpltdf['Actual'] = y
    xpltdf['Predicted'] = ypred
    xpltdf.to_csv(xyplt,index=False)
    print('Sucessfully generated xplot file %s'  % xyplt)

    if cmdlpredictionidcol:
        dfinpred = predictioncsv.addback_idcols(dfinpred)
        idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)


    savefiles(seisf = predictiondatacsv,
                sdf = dfinpred,
                sxydf = idtargetdf,
                outdir = cmdloutdir,
                ssuffix ='_KNN',
                name2merge =modeldatacsv)



def process_TuneCatBoostRegressor(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdlminmaxscale=None,
            cmdliterations=None,
            cmdllearningrate=None,
            cmdldepth=None,
            cmdlcv=None,
            cmdlscaleminmaxvalues =None,
            cmdlfeatureimportance=None,
            cmdloutdir=None,
            cmdlhideplot = False,
            cmdlvalsize=0.3):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file


    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    params ={'iterations': cmdliterations,
                'learning_rate': cmdllearningrate,
                'depth': cmdldepth}
    grdcv = GridSearchCV(CatBoostRegressor(loss_function= 'RMSE'),params,cv= cmdlcv)

    # Fit model
    grdcv.fit(X, y)
    print(grdcv.best_params_)
    clf = grdcv.best_estimator_
    print(grdcv.best_estimator_)
    # Get predictions
    ypred = clf.predict(X)

    msev = np.mean((ypred - y)**2)
    print('Metrics on Well data: ')
    print('Well Data Best Estimator MSE: %.4f' %(msev))
    r2v = r2_score(y,ypred)
    print('Well Data Best Estimator R2 : %10.3f' % r2v)




    pred = clf.predict(Xpred) #all seismic using optimum params

    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        #mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        dfinpred['CatBoostPred'] = predscaled
    else:
        dfinpred['CatBoostPred'] = pred

    #ax =plt.scatter(y,ypred)
    sns.set(color_codes=True)
    ax =sns.regplot(x=y,y=ypred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('CatBoostRegressor %s' %ycolname)
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_cbreg.pdf"
        xyplt = os.path.join(cmdloutdir,fname) +"_cbrxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_cbreg.pdf"
        xyplt = os.path.join(dirsplit,fname) +"_cbrxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    xpltcols =['Actual','Predicted']
    xpltdf = swa.iloc[:,:3].copy()  #copy well x y
    xpltdf['Actual'] = y
    xpltdf['Predicted'] = ypred
    xpltdf.to_csv(xyplt,index=False)
    print('Sucessfully generated xplot file %s'  % xyplt)


    if cmdlpredictionidcol:
        dfinpred = predictioncsv.addback_idcols(dfinpred)
        idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)


    savefiles(seisf = predictiondatacsv,
                sdf = dfinpred,
                sxydf = idtargetdf,
                outdir = cmdloutdir,
                ssuffix ='_stcbr',name2merge=modeldatacsv)




def process_feature_ranking(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdltestfeatures = None,
            cmdlcv = 3, #cv for SVR
            cmdltraintestsplit = .3,
            cmdlfeatures2keep =None,
            cmdllassoalpha = None):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data


    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    print(X.shape,len(colnames))


    if cmdltestfeatures == 'mutualinforeg':
        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        print ("Features sorted by their score:")
        # print (sorted(zip(map(lambda x: round(x, 4), mi),colnames), reverse=True))
        fimp = pd.DataFrame(sorted(zip(mi, colnames),reverse = True),columns=['MutualInfoRegression ','Attribute'])
        print('Feature Ranking by Mutual Info Regression: ')
        print(fimp)

    elif cmdltestfeatures =='rfe':
        #rank all features, i.e continue the elimination until the last one
        lm = LinearRegression()
        rfe = RFE(lm, n_features_to_select=cmdlfeatures2keep)
        rfe.fit(X,y)
        #print ("Features sorted by their rank:")
        #print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames)))

        scores = []
        for i in range(X.shape[1]):
             score = cross_val_score(lm, X[:, i:i+1], y, scoring="r2",
                                      cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
             #scores.append(round(np.mean(score), 3))
             scores.append(np.mean(score))
        #print (sorted(scores, reverse=True))
        r2fr = pd.DataFrame(sorted(zip(scores, colnames),reverse = True),columns=['R2 Score ','Attribute'])
        print('Feature Ranking by R2 scoring: ')
        print(r2fr)


    elif cmdltestfeatures =='svrcv':
        #rank all features, i.e continue the elimination until the last one

        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1,  cv = cmdlcv)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Cross Validated Recursive Feature Elimination Using SVR: ')
        print(fr)




    elif cmdltestfeatures =='svr':
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, cmdlfeatures2keep, step=1)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Recursive Feature Elimination Using SVR: ')
        print(fr)


    elif cmdltestfeatures =='rfregressor':
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        rf.fit(X,y)
        fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames),reverse = True),columns =['Importance','Attribute'])
        print(fi)
        #print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames)))
        #print(rf.feature_importances_)
        scores = []

        for i in range(X.shape[1]):
             score = cross_val_score(rf, X[:, i:i+1], y, scoring="r2",
                                      cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
             scores.append((round(np.mean(score), 3), colnames[i]))
        cvscoredf = pd.DataFrame(sorted( scores,reverse = True),columns =['Partial R2','Attribute'])
        print('\nCross Validation:')
        print (cvscoredf)

    elif cmdltestfeatures =='decisiontree':
        regressor = DecisionTreeRegressor(random_state=0)
        #cross_val_score(regressor, X, y, cv=3)
        regressor.fit(X,y)
        #print(regressor.feature_importances_)
        fr = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_), colnames),reverse = True),columns=['Importance','Attribute'])
        print('Feature Ranking with Decision Tree Regressor: ')
        print(fr)







#**********CatBoostRegressor
def process_CatBoostRegressor(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdlminmaxscale=None,
            cmdliterations=None,
            cmdllearningrate=None,
            cmdldepth=None,
            cmdlcv=None,
            cmdlscaleminmaxvalues =None,
            cmdlfeatureimportance=None,
            cmdloutdir=None,
            cmdlhideplot = False,
            cmdlvalsize=0.3):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file


    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)


    if cmdlfeatureimportance:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate= cmdllearningrate,
                    depth=cmdldepth,loss_function= 'RMSE',calc_feature_importance=True,random_seed=42)
        model.fit(X, y)
        fr = pd.DataFrame(sorted(zip(model.get_feature_importance(X,y), colnames),reverse=True),columns=['Importance','Attribute'])
        print('Feature Ranking with CatBoostRegressor: ')
        print(fr)




        plt.style.use('seaborn-whitegrid')
        ax=fr['Importance'].plot(kind='bar', figsize=(12,8));
        ax.set_xticklabels(fr['Attribute'],rotation = 45)
        ax.set_ylabel('Feature Importance')
        ax.set_title('CatBoostRegressor Feature Importance of %s' % cmdlwellattribcsv)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_cbrfi.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_cbrfi.pdf"
        fig = ax.get_figure()
        fig.savefig(pdfcl)


    elif cmdlcv:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate= cmdllearningrate,
                    depth=cmdldepth,loss_function= 'RMSE',random_seed=42)
        cvscore = cross_val_score(model,X,y,cv = cmdlcv,scoring='r2')
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" %(np.mean(cvscore[0])))
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv = cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" %(np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')


    else:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate= cmdllearningrate,
                depth=cmdldepth,loss_function= 'RMSE',random_seed=42)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' %(mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        pred = model.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' %(msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)


        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                        %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            #mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            dfinpred['CatBoostPred'] = predscaled
        else:
            dfinpred['CatBoostPred'] = pred

        #ax =plt.scatter(y,ypred)
        sns.set(color_codes=True)
        ax =sns.regplot(x=y,y=ypred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # plt.title('CatBoostRegressor %s' %dfin.columns[cmdlmodeltargetcol])
        plt.title('CatBoostRegressor %s' %ycolname)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_cbreg.pdf"
            xyplt = os.path.join(cmdloutdir,fname) +"_cbrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_cbreg.pdf"
            xyplt = os.path.join(dirsplit,fname) +"_cbrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        xpltcols =['Actual','Predicted']
        xpltdf = dfin[dfin.columns[0]].copy()  #copy back model id col
        xpltdf['Actual'] = y
        xpltdf['Predicted'] = ypred
        xpltdf.to_csv(xyplt,index=False)
        print('Sucessfully generated xplot file %s'  % xyplt)


        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)
            # idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = [4,5,-1])
            #This command is useful with classification after one hot encoder


        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_CBR',
                    name2merge=modeldatacsv)



#**********NuSVR support vector regresssion: uses nusvr
def process_NuSVR(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol= 0,    #id column for prediction csv
            cmdltargetscale = None,
            cmdlminmaxscale=None,
            cmdlnu=None,
            cmdlerrpenalty=None,
            cmdlcv=None,
            cmdlscaleminmaxvalues =None,
            cmdloutdir=None,
            cmdlhideplot = False,
            cmdlvalsize=0.3):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file


    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)




    if cmdlcv:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        cvscore = cross_val_score(model,X,y,cv = cmdlcv,scoring='r2')
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" %(np.mean(cvscore[0])))
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv = cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" %(np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')


    else:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' %(mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        pred = model.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' %(msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)


        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                        %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            #mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            dfinpred['NuSVRPred'] = predscaled
        else:
            dfinpred['NuSVRPred'] = pred

        #ax =plt.scatter(y,ypred)
        sns.set(color_codes=True)
        ax =sns.regplot(x=y,y=ypred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # plt.title('CatBoostRegressor %s' %dfin.columns[cmdlmodeltargetcol])
        plt.title('NuSVR %s' %ycolname)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_nusvr.pdf"
            xyplt = os.path.join(cmdloutdir,fname) +"_nusvrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_nusvr.pdf"
            xyplt = os.path.join(dirsplit,fname) +"_nusvrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        xpltcols =['Actual','Predicted']
        xpltdf = dfin[dfin.columns[0]].copy()  #copy back model id col
        xpltdf['Actual'] = y
        xpltdf['Predicted'] = ypred
        xpltdf.to_csv(xyplt,index=False)
        print('Sucessfully generated xplot file %s'  % xyplt)


        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)
            # idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = [4,5,-1])
            #This command is useful with classification after one hot encoder


        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_NuSVR',
                    name2merge=modeldatacsv)





#**********ANNRegressor
def process_ANNRegressor(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdlminmaxscale=None,
            cmdlnodes=None,  #same number as num layers
            cmdlactivation=None, #same numberof codes as num layers
            cmdlepochs = None, #one number
            cmdlbatch = None,  #one number
            cmdlcv=None,
            cmdlscaleminmaxvalues =None,
            cmdloutdir=None,
            cmdlhideplot = False,
            cmdlvalsize=0.3,
            cmdlradius= None
            ):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolname = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    #targetcol has to be None
    #scalesave has to be False to read from saved modeldata scale
    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file


    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    cmdllayers = len(cmdlnodes)
    def build_model():
        indim = cmdlmodelcolsrange[1] - cmdlmodelcolsrange[0] +1
        model = Sequential()
        model.add(Dense(cmdlnodes[0], input_dim=indim, kernel_initializer= 'normal' , activation= cmdlactivation[0] ))
        for i in range(1,cmdllayers):
            model.add(Dense(cmdlnodes[i], kernel_initializer= 'normal' , activation= cmdlactivation[i] ))
        model.add(Dense(1, kernel_initializer= 'normal' ))
        # Compile model
        model.compile (loss= 'mean_squared_error' , optimizer= 'adam')
        return model


    if cmdlcv:
        kfold = KFold(n_splits=cmdlcv, random_state=42)
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        cvscore = cross_val_score(estimator,X,y,cv = cmdlcv,scoring='r2')
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" %(np.mean(cvscore[0])))
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv = cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" %(np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')


    else:
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        estimator.fit(X, y )
        # Get predictions
        ypred = estimator.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' %(mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        pred = estimator.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        estimator.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))

        yvalpred = estimator.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' %(msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)


        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                        %(cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            #mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            dfinpred['ANNPred'] = predscaled
        else:
            dfinpred['ANNPred'] = pred

        #ax =plt.scatter(y,ypred)
        sns.set(color_codes=True)
        ax =sns.regplot(x=y,y=ypred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # plt.title('CatBoostRegressor %s' %dfin.columns[cmdlmodeltargetcol])
        plt.title('CatBoostRegressor %s' %ycolname)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_annreg.pdf"
            xyplt = os.path.join(cmdloutdir,fname) +"_annrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_annreg.pdf"
            xyplt = os.path.join(dirsplit,fname) +"_annrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        xpltcols =['Actual','Predicted']
        xpltdf = dfin[dfin.columns[0]].copy()  #copy back model id col
        xpltdf['Actual'] = y
        xpltdf['Predicted'] = ypred
        xpltdf.to_csv(xyplt,index=False)
        print('Sucessfully generated xplot file %s'  % xyplt)

        """

        #map back interpolate
        xs = dfinpred.iloc[:,cmdlpredictionidcol[0]].values
        ys = dfinpred.iloc[:,cmdlpredictionidcol[1]].values
        xys = np.transpose(np.vstack((xs,ys)))
        xw = dfin.iloc[:,cmdlmodelidcol[1]].values
        yw = dfin.iloc[:,cmdlmodelidcol[2]].values
        xyw = np.transpose(np.vstack((xw,yw)))

        print('******Map Back Interpolation')
        zw = map2ddata(xys,pred,xyw,cmdlradius,'idw')

        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        #plotting model predicted values
        fig = plt.figure()
        # fig.suptitle( ' Model vs Map Prediction ' )
        fig.suptitle( ' Model vs Map Prediction ' )
        ax = fig.add_subplot(111)
        qc0 = np.polyfit(y,ypred,1)
        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='b',lw=2)

        #plot map predicted values
        fig.subplots_adjust(top=0.9)
        qc1 = np.polyfit(y,zw,1)
        yvi1 = np.polyval(qc1,xvi)
        plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
        plt.plot(xvi,yvi1,c='r',lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        ytitle = 'Predicted'
        dp = 3

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
                (2,qc0[0],2,qc0[1]),xy =(xvi[0],yvi0[0]),xytext=(0.14,0.85),
                textcoords='figure fraction', fontsize=10)
        ax.annotate('Map = %-.*f * Actual + %-.*f' %
                (2,qc1[0],2,qc1[1]),xy =(xvi[0],yvi1[0]),xytext=(0.14,0.81),
                textcoords='figure fraction', fontsize=10)



        plt.legend('lower right')
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl2 = os.path.join(cmdloutdir,fname) +"_annreg2.pdf"
            xyplt2 = os.path.join(cmdloutdir,fname) +"_annrxplt2.csv"
        else:
            pdfcl2 = os.path.join(dirsplit,fname) +"_annreg2.pdf"
            xyplt2 = os.path.join(dirsplit,fname) +"_annrxplt2.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl2)
        print('Model Predicted Line:',qc0)
        print('Map Predicted Line:',qc1)
        xpltcols1 =['Actual','ModelPredicted','MapPredicted']
        xpltdf1 = swa.iloc[:,:3].copy()  #copy well x y
        xpltdf1['Actual'] = y
        xpltdf1['ModelPredicted'] = ypred
        xpltdf1['MapPredicted'] =zw
        xpltdf1.to_csv(xyplt2,index=False)
        print('Sucessfully generated xplot file %s'  % xyplt2)

        """

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = -1)
            # idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = [4,5,-1])
            #This command is useful with classification after one hot encoder


        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_ANNR',
                    name2merge=modeldatacsv)





def process_CatBoostClassifier(modeldatacsv,predictiondatacsv,
                            cmdlmodelcolsrange=None,
                            cmdlmodelcolselect = None,
                            cmdlmodeltargetcol=None,
                            cmdlmodelidcol = None,
                            cmdlsamplemodel=None,
                            cmdlmodelscalefeatures = None,
                            cmdlmodelscalesave = True,
                            cmdlpredictioncolsrange=None,
                            cmdlpredictioncolselect = None,
                            cmdlpredictionscalefeatures = None,
                            cmdlsampleprediction=None,
                            cmdlpredictionscalesave = None,
                            cmdlkind = None,
                            cmdlpredictionidcol = None,
                            cmdlqcut=None,
                            cmdlnqcutclasses = 3,
                            cmdltargetencode=None,
                            cmdlcoded = None,
                            cmdloutdir=None,
                            cmdlfeatureimportance = None,
                            cmdliterations = None,
                            cmdllearningrate=None,
                            cmdldepth = None,
                            cmdlcv=None,
                            cmdlvalsize=0.3,
                            cmdlhideplot=False,
                            cmdlclassweight=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)




    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
    print('Pred df:',dfinpred.shape)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)



    if cmdlfeatureimportance:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate= cmdllearningrate,
                    depth=cmdldepth,loss_function= 'MultiClass',calc_feature_importance=True,
                    random_seed=42)
        clf.fit(X, y)
        fr = pd.DataFrame(sorted(zip(clf.get_feature_importance(X,y), colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with CatBoostClassifier: ')
        print(fr)


    elif cmdlcv:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate= cmdllearningrate,
                    depth=cmdldepth,loss_function= 'MultiClass',random_seed=42)
        cvscore = cross_val_score(clf,X,y,cv = cmdlcv)
        print ( "Accuracy: %.3f%% (%.3f%%)"  % (cvscore.mean()*100.0, cvscore.std()*100.0))
        #print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        #print("Mean Score: %10.4f" %(np.mean(cvscore)))
        print('No files will be generated. Re-run without cross validation')


    else:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate= cmdllearningrate,
                    depth=cmdldepth,loss_function= 'MultiClass',random_seed=42)
        clf.fit(X, y)
        # Get predictions
        y_clf = clf.predict(Xpred, prediction_type='Class')
        #ypred = clf.predict(Xpred, prediction_type='RawFormulaVal')
        allprob = clf.predict_proba(Xpred)
        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        clf.fit(Xtrain, ytrain)

        yvalpred = clf.predict(Xval)
        yvalproba = clf.predict_proba(Xval)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))
        print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
        print('Validation Data Log Loss: %10.4f' %log_loss(yval,yvalproba))
        ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
        print(pd.crosstab(ydf['A'],ydf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        #print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
        print(classification_report(yval.ravel(),yvalpred.ravel()))



        #[(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        #add class column before probabilities
        dfinpred['CatBoost'] = y_clf
        for i in range(len(ycolnames)):
            dfinpred[ycolnames[i]] = allprob[:,i]

        for i in range(len(ycolnames)):
            yw_prob = clf.predict_proba(X)[:,i]
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) +"_cbcroc%1d.pdf" %i
            else:
                pdfcl = os.path.join(dirsplit,fname) +"_cbcroc%1d.pdf" %i
            plot_roc_curve(y,yw_prob,i,cmdlhideplot,pdfcl)


        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d'% len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))





        dfin['predqcodes'] = yw

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)


        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    wellf = modeldatacsv,
                    sxydf = idtargetdf,
                    wdf = dfin,
                    outdir = cmdloutdir,
                    ssuffix ='_scbc',
                    wsuffix = '_wcbc',name2merge=modeldatacsv)




#***************************
def process_TuneCatBoostClassifier(modeldatacsv,predictiondatacsv,
                            cmdlmodelcolsrange=None,
                            cmdlmodelcolselect = None,
                            cmdlmodeltargetcol=None,
                            cmdlmodelidcol = None,
                            cmdlsamplemodel=None,
                            cmdlmodelscalefeatures = None,
                            cmdlmodelscalesave = True,
                            cmdlpredictioncolsrange=None,
                            cmdlpredictioncolselect = None,
                            cmdlpredictionscalefeatures = None,
                            cmdlsampleprediction=None,
                            cmdlpredictionscalesave = None,
                            cmdlkind = None,
                            cmdlpredictionidcol = None,
                            cmdlqcut=None,
                            cmdlnqcutclasses = 3,
                            cmdltargetencode=None,
                            cmdlcoded = None,
                            cmdloutdir=None,
                            cmdliterations = None,
                            cmdllearningrate=None,
                            cmdldepth = None,
                            cmdlcv=None,
                            cmdlvalsize=0.3,
                            cmdlhideplot=False,
                            cmdlclassweight=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)




    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
    print('Pred df:',dfinpred.shape)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)


    params ={'iterations': cmdliterations,
                'learning_rate': cmdllearningrate,
                'depth': cmdldepth}
    grdcv = GridSearchCV(CatBoostClassifier(loss_function= 'MultiClass'),params,cv= cmdlcv)

    # Fit model
    grdcv.fit(X, y)
    print(grdcv.best_params_)
    clf = grdcv.best_estimator_
    # Get predictions
    wpred = clf.predict(X)
    y_clf = clf.predict(Xpred, prediction_type='Class')
    #ypred = clf.predict(Xpred, prediction_type='RawFormulaVal')
    allprob = clf.predict_proba(Xpred)
    wproba = clf.predict_proba(X)
    print('All Data Accuracy Score: %10.4f' % accuracy_score(y,wpred))
    print('Log Loss: %10.4f' %log_loss(y,wproba))

    #[(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
    #add class column before probabilities
    dfinpred['TunedCatBoost'] = y_clf
    for i in range(len(ycolnames)):
        dfinpred[ycolnames[i]] = allprob[:,i]

    yw = clf.predict(X)

    ywproba = clf.predict_proba(X)
    print('Full Data size: %5d'% len(yw))
    print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
    print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
    ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
    print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
    print(classification_report(y.ravel(),yw.ravel()))





    dfin['predqcodes'] = yw


    if cmdlpredictionidcol:
        dfinpred = predictioncsv.addback_idcols(dfinpred)
        probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
        idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)



    savefiles(seisf = predictiondatacsv,
                sdf = dfinpred,
                wellf = modeldatacsv,
                wdf = dfin,
                sxydf = idtargetdf,
                outdir = cmdloutdir,
                ssuffix ='_stcbc',
                wsuffix = '_wtcbc',name2merge=modeldatacsv)



#***************************
def process_logisticreg(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdltargetencode=None,
            cmdlqcut=None,
            cmdlnqcutclasses=3,
            cmdlcoded=None,
            cmdlloadencoder=False,
            cmdlclassweight=False,
            cmdlcv=None,
            cmdlvalsize=0.3,
            cmdloutdir=None,
            cmdlhideplot=False):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
    print('Pred df:',dfinpred.shape)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)



    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        results = cross_val_score(clf, X, y, cv=kfold)
        print ( "Logistic Regression Accuracy: %.3f%% (%.3f%%)"  % (results.mean()*100.0, results.std()*100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        clf.fit(X,y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        clf.fit(Xtrain, ytrain)

        yvalpred = clf.predict(Xval)
        yvalproba = clf.predict_proba(Xval)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))
        print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
        print('Validation Data Log Loss: %10.4f' %log_loss(yval,yvalproba))
        ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
        print(pd.crosstab(ydf['A'],ydf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        #print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
        print(classification_report(yval.ravel(),yvalpred.ravel()))

        #[(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        #add class column before probabilities


        dfinpred['LRClass'] = y_clf
        for i in range(len(ycolnames)):
            dfinpred[ycolnames[i]] = allprob[:,i]

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)

        dirsplit,fextsplit= os.path.split(modeldatacsv)
        fname,fextn= os.path.splitext(fextsplit)

        for i in range(len(ycolnames)):
            yw_prob = clf.predict_proba(X)[:,i]
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) +"_lgrroc%1d.pdf" %i
            else:
                pdfcl = os.path.join(dirsplit,fname) +"_lgrroc%1d.pdf" %i
            plot_roc_curve(y,yw_prob,i,cmdlhideplot,pdfcl)


        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d'% len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))

        dfin['predqcodes'] = yw
        #pdfbar = os.path.join(dirsplit,fname) +"_lgrbar.pdf"
        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    wellf = modeldatacsv,
                    wdf = dfin,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_slgrg',
                    wsuffix = '_wlgrg',name2merge=modeldatacsv)



#***************************
def process_GaussianNaiveBayes(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdltargetencode=None,
            cmdlqcut=None,
            cmdlnqcutclasses=3,
            cmdlcoded=None,
            cmdlloadencoder=False,
            cmdlclassweight=False,
            cmdlcv=None,
            cmdlvalsize=0.3,
            cmdloutdir=None,
            cmdlhideplot=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
    print('Pred df:',dfinpred.shape)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)

        #need to check if there is a class weight option for GaussianNB
        clf = GaussianNB()
        results = cross_val_score(clf, X, y, cv=kfold)
        print ( "Gaussian Naive Bayes Accuracy: %.3f%% (%.3f%%)"  % (results.mean()*100.0, results.std()*100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = GaussianNB()
        clf.fit(X,y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        clf.fit(Xtrain, ytrain)

        yvalpred = clf.predict(Xval)
        yvalproba = clf.predict_proba(Xval)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))
        print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
        print('Validation Data Log Loss: %10.4f' %log_loss(yval,yvalproba))
        ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
        print(pd.crosstab(ydf['A'],ydf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        #print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
        print(classification_report(yval.ravel(),yvalpred.ravel()))


        #add class column before probabilities
        dfinpred['GNBClass'] = y_clf

        #[(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(len(ycolnames)):
            dfinpred[ycolnames[i]] = allprob[:,i]



        # dirsplit,fextsplit= os.path.split(modeldatacsv)
        # fname,fextn= os.path.splitext(fextsplit)

        for i in range(len(ycolnames)):
            yw_prob = clf.predict_proba(X)[:,i]
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) +"_gnbroc%1d.pdf" %i
            else:
                pdfcl = os.path.join(dirsplit,fname) +"_gnbroc%1d.pdf" %i
            plot_roc_curve(y,yw_prob,i,cmdlhideplot,pdfcl)

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d'% len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))





        dfin['predqcodes'] = yw
        #pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)

        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    wellf = modeldatacsv,
                    wdf = dfin,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_sgnb',
                    wsuffix = '_wgnb',name2merge=modeldatacsv)



#***************************
def process_QuadraticDiscriminantAnalysis(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdltargetencode=None,
            cmdlqcut=None,
            cmdlnqcutclasses=3,
            cmdlcoded=None,
            cmdlloadencoder=False,
            cmdlclassweight=False,
            cmdlcv=None,
            cmdlvalsize=0.3,
            cmdloutdir=None,
            cmdlhideplot=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
        colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
        scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
        scalesave=False,sample=cmdlsampleprediction)
        #I hard coded prediction scale save to false to read already saved scaler file

    #extract and scale columns of data only
    Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
    print('Pred df:',dfinpred.shape)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)

        #need to check if there is a class weight option for GaussianNB
        # clf = QDA()
        clf = QuadraticDiscriminantAnalysis()
        results = cross_val_score(clf, X, y, cv=kfold)
        print ( "Quadratic Discriminant Analysis Accuracy: %.3f%% (%.3f%%)"  % (results.mean()*100.0, results.std()*100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        # clf = QDA()
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X,y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        clf.fit(Xtrain, ytrain)

        yvalpred = clf.predict(Xval)
        yvalproba = clf.predict_proba(Xval)
        print('Train Data size: %5d, Validation Data size: %5d'% (len(ytrain),len(yval)))
        print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
        print('Validation Data Log Loss: %10.4f' %log_loss(yval,yvalproba))
        ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
        print(pd.crosstab(ydf['A'],ydf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        #print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
        print(classification_report(yval.ravel(),yvalpred.ravel()))


        #add class column before probabilities
        dfinpred['QDAClass'] = y_clf

        #[(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(len(ycolnames)):
            dfinpred[ycolnames[i]] = allprob[:,i]



        # dirsplit,fextsplit= os.path.split(modeldatacsv)
        # fname,fextn= os.path.splitext(fextsplit)

        for i in range(len(ycolnames)):
            yw_prob = clf.predict_proba(X)[:,i]
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) +"_qdaroc%1d.pdf" %i
            else:
                pdfcl = os.path.join(dirsplit,fname) +"_qdaroc%1d.pdf" %i
            plot_roc_curve(y,yw_prob,i,cmdlhideplot,pdfcl)

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d'% len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))





        dfin['predqcodes'] = yw
        #pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)

        savefiles(seisf = predictiondatacsv,
                    sdf = dfinpred,
                    wellf = modeldatacsv,
                    wdf = dfin,
                    sxydf = idtargetdf,
                    outdir = cmdloutdir,
                    ssuffix ='_sqda',
                    wsuffix = '_wqda',name2merge=modeldatacsv)








def process_testCmodels(modeldatacsv,predictiondatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdltargetencode=None,
            cmdlqcut=None,
            cmdlnqcutclasses=3,
            cmdlcoded=None,
            cmdlcv=None,
            cmdlvalsize=0.3,
            cmdloutdir=None,
            cmdlhideplot=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


        models = []
        models.append(( ' LR ' , LogisticRegression()))
        models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
        models.append(( ' KNN ' , KNeighborsClassifier()))
        models.append(( ' CART ' , DecisionTreeClassifier()))
        models.append(( ' NB ' , GaussianNB()))
        models.append(( ' SVM ' , SVC()))
        # evaluate each model in turn
        results = []
        names = []
        resultsmean = []
        resultsstd = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = KFold(n_splits=cmdl.cv, random_state=7)
            cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "Model: %s: Mean Accuracy: %0.4f Std: (%0.4f)" % (name, cv_results.mean(), cv_results.std())
            #print (msg)
            resultsmean.append(cv_results.mean())
            resultsstd.append(cv_results.std())
        modeltest = pd.DataFrame(list(zip(names,resultsmean,resultsstd)),columns=['Model','Model Mean Accuracy','Accuracy STD'])
        print(modeltest)

        dirsplit,fextsplit= os.path.split(modeldatacsv)
        fname,fextn= os.path.splitext(fextsplit)
        if cmdl.outdir:
            pdfcl = os.path.join(cmdl.outdir,fname) +"_testcm.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_testcm.pdf"


        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle( ' Classification Algorithm Comparison ' )
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig(pdfcl)
        if not cmdl.hideplot:
            plt.show()



#***************************
def process_clustertest(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdloutdir=None,
            cmdlhideplot=False):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        targetcol=cmdlmodeltargetcol,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_cla.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_cla.pdf"
    inertia = list()
    delta_inertia = list()
    for k in range(1,21):
        clustering = KMeans(n_clusters = k, n_init = 10,random_state = 1)
        clustering.fit(X)
        if inertia:
            delta_inertia.append(inertia[-1] - clustering.inertia_)
        inertia.append(clustering.inertia_)
    with PdfPages(pdfcl) as pdf:
        plt.figure(figsize=(8,8))
        plt.plot([k for k in range(2,21)], delta_inertia,'ko-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Rate of Change of Intertia')
        plt.title('KMeans Cluster Analysis')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()

        #print('Successfully generated %s file'  % pdfcl)



#***************************
def process_clustering(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlnclusters=None,
            cmdlplotsilhouette=False,
            cmdladdclass=None,
            cmdloutdir=None,
            cmdlhideplot=False):

    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        targetcol=cmdlmodeltargetcol,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)


    clustering = KMeans(n_clusters = cmdlnclusters,
        n_init = 5,
        max_iter = 300,
        tol = 1e-04,
        random_state = 1)
    ylabels = clustering.fit_predict(X)
    nlabels = np.unique(ylabels)
    print('nlabels',nlabels)


    if cmdladdclass == 'labels':
        dfin['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        dfin = pd.concat([dfin,classdummies],axis = 1)
    print(dfin.shape)


    dfintxt = dfin.copy()
    if cmdladdclass == 'labels':
        dfintxt['Class']= ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        dfintxt = pd.concat([dfintxt,classdummies],axis = 1)

    if cmdladdclass == 'labels':
        savefiles(seisf = modeldatacsv,
                    sdf = dfin,
                    sxydf = dfintxt,
                    outdir = cmdloutdir,
                    ssuffix ='_cl')
    else:
        savefiles(seisf = modeldatacsv,
                    sdf = dfin,
                    sxydf = dfintxt,
                    outdir = cmdloutdir,
                    ssuffix ='_cld')


    '''
    Warning: Do not use sample to enable plot silhouette and add labels or dummies
    Better make a seperate run for silhouette plot on sampled data then use full data
    to add labels
    '''



    if cmdlplotsilhouette:
        dirsplit,fextsplit= os.path.split(modeldatacsv)
        fname,fextn= os.path.splitext(fextsplit)
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_silcl.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_silcl.pdf"
        #only resample data if plotting silhouette
        ylabels = clustering.fit_predict(dfin)
        n_clusters = ylabels.shape[0]
        silhouette_vals = silhouette_samples(dfin, ylabels, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(nlabels):
            c_silhouette_vals = silhouette_vals[ylabels == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(i / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)
            silhouette_avg = np.mean(silhouette_vals)
            plt.axvline(silhouette_avg,
            color="red",
            linestyle="--")
            plt.yticks(yticks, ylabels + 1)
            plt.ylabel('Cluster')
            plt.xlabel('Silhouette coefficient')
        plt.savefig(pdfcl)
        if not cmdlhideplot:
            plt.show()



#***************************
def process_GaussianMixtureModel(modeldatacsv,
            predictiondatacsv = None,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlsampleprediction=None,   #sampling of prediction data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlpredictioncolsrange = None,
            cmdlpredictioncolselect = None,
            cmdlpredictionscalefeatures=True, #scale predcition data
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdlpredictionscalesave=False,    #save prediction scaler should be false to use already saved scaler
            cmdlkind='standard',
            cmdlpredictionidcol=None,    #id column for prediction csv
            cmdltargetscale = None,
            cmdltargetencode=None,
            cmdlqcut=None,
            cmdlnqcutclasses=3,
            cmdlcoded=None,
            cmdlloadencoder=False,
            cmdlmodelbayesian=False,
            cmdlmodelncomponents=4,
            cmdloutdir=None,
            cmdlhideplot=False):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,targetcol=cmdlmodeltargetcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        qcut=cmdlqcut,nqcutclasses=cmdlnqcutclasses,
        targetencode=cmdltargetencode,coded=cmdlcoded,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    # returns X data, column names, and dataframe that is scaled
    X,colnames,dfin= modelcsv.extract_scale_cols()
    y,ycolnames = modelcsv.extract_target()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)


    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)



    if cmdlmodelbayesian:
        gmm = mixture.BayesianGaussianMixture(n_components=cmdlmodelncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)
        pltname ='bayes'
    else:
        gmm = mixture.GaussianMixture(n_components=cmdlmodelncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)

    probaclassnames = ['GMMClass%d'%i for i in range(cmdlmodelncomponents)]

    xpdf = np.linspace(-4, 3, 1000)
    _,ax = plt.subplots()
    for i in range(gmm.n_components):
        pdf = gmm.weights_[i] * sts.norm(gmm.means_[i, 0],np.sqrt(gmm.covariances_[i])).pdf(xpdf)
        ax.fill(xpdf, pdf, edgecolor='none', alpha=0.3,label='%s'%probaclassnames[i])
    ax.legend()
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) +"_gmm%d%s.pdf" %(cmdlncomponents,pltname)
    else:
        pdfcl = os.path.join(dirsplit,fname) +"_gmm%d%s.pdf" %(cmdlncomponents,pltname)
    fig = ax.get_figure()
    fig.savefig(pdfcl)


    if predictiondatacsv:
        predictioncsv = prepcsv(predictiondatacsv,idcols=cmdlpredictionidcol,targetcol=None,
            colsrange=cmdlpredictioncolsrange,colsselect = cmdlpredictioncolselect,
            scalefeatures = cmdlpredictionscalefeatures,scaletype= cmdlkind,
            scalesave=False,sample=cmdlsampleprediction)
            #I hard coded prediction scale save to false to read already saved scaler file

        #extract and scale columns of data only
        Xpred,colnamespred,dfinpred = predictioncsv.extract_scale_cols()
        print('Pred df:',dfinpred.shape)

        y_clf = gmm.predict(Xpred)
        allprob = gmm.predict_proba(Xpred)


        dfinpred['GMMClass'] = y_clf
        for i in range(len(ycolnames)):
            dfinpred[ycolnames[i]] = allprob[:,i]

        if cmdlpredictionidcol:
            dfinpred = predictioncsv.addback_idcols(dfinpred)
            probacolnums = dfinpred.columns[-len(ycolnames):].tolist()
            idtargetdf =predictioncsv.idtarget_merge(predicteddf = dfinpred,predicteddfcols = probacolunms)

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)

    for i in range(len(ycolnames)):
        yw_prob = clf.predict_proba(X)[:,i]
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) +"_gmmroc%1d.pdf" %i
        else:
            pdfcl = os.path.join(dirsplit,fname) +"_gmmroc%1d.pdf" %i
        plot_roc_curve(y,yw_prob,i,cmdlhideplot,pdfcl)


    yw = clf.predict(X)
    ywproba = clf.predict_proba(X)
    print('Full Data size: %5d'% len(yw))
    print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
    print('Full Data Log Loss: %10.4f' %log_loss(y,ywproba))
    ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
    print(pd.crosstab(ywdf['A'],ywdf['P'],rownames = ['Actuall'], colnames =['Predicted']))
    print(classification_report(y.ravel(),yw.ravel()))

    dfin['predqcodes'] = yw
    #pdfbar = os.path.join(dirsplit,fname) +"_lgrbar.pdf"
    savefiles(seisf = predictiondatacsv,
                sdf = dfinpred,
                wellf = modeldatacsv,
                wdf = dfin,
                sxydf = idtargetdf,
                outdir = cmdloutdir,
                ssuffix ='_sgmm',
                wsuffix = '_wgmm',name2merge=modeldatacsv)






def process_dbscan(modeldatacsv,
            cmdlmodelcolsrange = None,
            cmdlmodelcolselect = None,
            cmdlmodelidcol=None,     #idcolumn for model csv
            cmdlmodeltargetcol=None,    #target column for model csv
            cmdlsamplemodel=None,   #sampling of model data
            cmdlmodelscalefeatures=True, #scale model data
            cmdlkind='standard',
            cmdlmodelscalesave=True,    #save model scaler should be true
            cmdleps=None,
            cmdlminsamples=10,
            cmdladdclass=None,
            cmdloutdir=None,
            cmdlhideplot=False):


    modelcsv = prepcsv(modeldatacsv,idcols=cmdlmodelidcol,
        colsrange=cmdlmodelcolsrange,colsselect = cmdlmodelcolselect,
        targetcol=cmdlmodeltargetcol,
        scalefeatures = cmdlmodelscalefeatures,scaletype= cmdlkind,
        scalesave=True,sample=cmdlsamplemodel)
        #I hard coded scalesave to True for model data

    #extract and scale columns of data only
    X,colnames,dfin= modelcsv.extract_scale_cols()
    if cmdlmodelidcol:
        dfin = modelcsv.addback_idcols(dfin)
    if cmdlmodeltargetcol:
        dfin,tcolnum = modelcsv.addback_targetcol(dfin)

    dirsplit,fextsplit= os.path.split(modeldatacsv)
    fname,fextn= os.path.splitext(fextsplit)


    dbscan=DBSCAN(eps=cmdleps, metric='euclidean', min_samples=cmdlminsamples)
    ylabels = dbscan.fit_predict(X)
    print('Labels count per class:',list(Counter(ylabels).items()))

    #n_clusters = len(set(ylabels)) - (1 if -1 in ylabels else 0)
    n_clusters = len(set(ylabels))
    print('Estimated number of clusters: %d' % n_clusters)

    if cmdladdclass == 'labels':
        dfin['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        dfin = pd.concat([dfin,classdummies],axis = 1)
    print(dfin.shape)


    dfintxt = dfin.copy()
    if cmdladdclass == 'labels':
        dfintxt['Class']= ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        dfintxt = pd.concat([dfintxt,classdummies],axis = 1)

    if cmdladdclass == 'labels':
        savefiles(seisf = modeldatacsv,
                    sdf = dfin,
                    sxydf = dfintxt,
                    outdir = cmdloutdir,
                    ssuffix ='_dbscn')
    else:
        savefiles(seisf = modeldatacsv,
                    sdf = dfin,
                    sxydf = dfintxt,
                    outdir = cmdloutdir,
                    ssuffix ='_dbscnd')





def process_semisupervised(wfname,sfname,cmdlwcolsrange=None,
                cmdlwtargetcol=None,cmdlwellsxyzcols=None,
                cmdlsample=0.005,cmdlkernel='knn',cmdlnneighbors=7,
                cmdlcol2drop=None,cmdloutdir=None):
    i4w = pd.read_csv(wfname)
    if cmdlcol2drop:
        i4w.drop(i4w.columns[cmdlcol2drop],axis=1,inplace=True)
    dirsplitw,fextsplit= os.path.split(wfname)
    fnamew,fextn= os.path.splitext(fextsplit)
    if cmdloutdir:
        ppsave = os.path.join(cmdloutdir,fnamew) +"_pw.csv"
    else:
        ppsave = os.path.join(dirsplitw,fnamew) +"_pw.csv"
    # print('target col',cmdlwtargetcol)
    # print(i4w[i4w.columns[cmdlwtargetcol]],i4w.columns[cmdlwtargetcol])
    # coln = i4w.columns[cmdlwtargetcol]
    # print('coln:',coln)
    if cmdlcol2drop:
        cmdlwtargetcol -=1
    i4w['qa'],qbins = pd.qcut(i4w[i4w.columns[cmdlwtargetcol]],3,labels=['Low','Medium','High'],retbins=True)

    i4w['qcodes'] = i4w['qa'].cat.codes
    print('codes: ',i4w['qcodes'].unique())

    i4s = pd.read_csv(sfname)

    #i4w.drop(['Av_PHIT', 'qa'],axis=1,inplace=True)
    i4w.drop(i4w.columns[[cmdlwtargetcol,cmdlwtargetcol+1]],axis=1,inplace=True)
    i4sx = i4s.sample(frac=cmdlsample,random_state=42)
    i4sxi =i4sx.reset_index()
    i4sxi.drop('index',axis=1,inplace=True)
    i4sxi['Well1']= ['PW%d'%i for i in i4sxi.index]
    i4sxi.insert(0,'Well',i4sxi['Well1'])
    i4sxi.drop('Well1',axis=1,inplace=True)
    i4sxi['qcodes']=[(-1) for i in i4sxi.index]
    wcols = list(i4w.columns)
    i4sxi.columns=wcols
    i4 = pd.concat([i4w,i4sxi],axis=0)
    X = i4[i4.columns[cmdlwcolsrange[0] : cmdlwcolsrange[1] + 1]].values
    y = i4[i4.columns[cmdlwtargetcol]].values
    print(Counter(list(y)).items())
    lblsprd= LabelSpreading(kernel=cmdlkernel,n_neighbors= cmdlnneighbors)
    lblsprd.fit(X,y)
    ynew = lblsprd.predict(X)
    print(Counter(list(ynew)).items())
    i4['qcodespred']=ynew
    i4.drop(i4.columns[cmdlwtargetcol],axis=1,inplace=True)
    i4.to_csv(ppsave,index=False)
    print('Successfully generated %s file'  % ppsave)
    i4xy = i4.copy()
    i4xy.drop(i4xy.columns[cmdlwcolsrange[0] : cmdlwcolsrange[1]+1],axis=1,inplace=True)
    if cmdloutdir:
        ppxysave = os.path.join(cmdloutdir,fnamew) + "_pw.txt"
    else:
        ppxysave = os.path.join(dirsplitw,fnamew) + "_pw.txt"
    i4xy.to_csv(ppxysave,sep=' ',index=False)
    print('Successfully generated %s file'  % ppxysave)



def cmnt(line):
        if '#' in line:
            return True
        else:
            return False


class   ClassificationMetrics:
    '''
    Compute Classification Metrics
    '''
    def __init__(self,actual,predicted,tolist=True,tocsv=False):
        self.actual =actual
        self.predicted = predicted
        self.tolist = tolist
        self.tocsv = tocsv

    def confusion(self):
        if self.tolist:
            print('Confusion Report: ')
            print(pd.crosstab(self.actual,self.predicted,rownames = ['Actual'], colnames =['Predicted']))

    def accuracy(self):
        if self.tolist:
            print('Accuracy Score: ',accuracy_score(self.actual,self.predicted))

    def clfreport(self):
        if self.tolist:
            print('Classification Report: ')
            print(classification_report(self.actual,self.predicted))


def getcommandline(*oneline):
    allcommands =['workflow','dropcols','listcsvcols','featurescale','target2code',
                'mergeidtarget','PCAanalysis','PCAfilter','scattermatrix',
                'qclin','linreg','featureranking','linfitpredict','KNNtest','KNNfitpredict','CatBoostRegressor',
                'CatBoostClassifier','testCmodels','logisticreg','GaussianNaiveBayes','clustertest','clustering',
                'tSNE','TuneCatBoostClassifier','TuneCatBoostRegressor','DBSCAN','targetscale','semisupervised',
                'ANNRegressor','QDA','NuSVR','GaussianMixtureModel']

    mainparser = argparse.ArgumentParser(description='Seismic ánd Well Attributes Modeling.')
    mainparser.set_defaults(which=None)
    subparser = mainparser.add_subparsers(help='File name listing all attribute grids')

    wrkflowparser = subparser.add_parser('workflow',help='Workflow file instead of manual steps')
    wrkflowparser.set_defaults(which='workflow')
    wrkflowparser.add_argument('commandfile',help='File listing workflow')
    wrkflowparser.add_argument('--startline',type=int,default =0,help='Line in file to start flow from. default=0')
    #wrkflowparser.add_argument('--stopat',type=int,default =None,help='Line in file to end flow after. default=none')


    #*************dropcols from csv
    dropparser = subparser.add_parser('dropcols',help='csv drop columns')
    dropparser.set_defaults(which = 'dropcols')
    dropparser.add_argument('csvfile',help='csv file to drop columns')
    dropparser.add_argument('--cols2drop',type=int,nargs='*',default=None,help='default=none')
    dropparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************listcsvcols
    listcolparser = subparser.add_parser('listcsvcols',help='List header row of any csv')
    listcolparser.set_defaults(which ='listcsvcols')
    listcolparser.add_argument('csvfile',help='csv file name')




    #*************PCA analysis
    pcaaparser = subparser.add_parser('PCAanalysis',help='PCA analysis')
    pcaaparser.set_defaults(which='PCAanalysis')
    pcaaparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    pcaaparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcaaparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    pcaaparser.add_argument('--modelidcol',type=int,nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    pcaaparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    pcaaparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    pcaaparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    pcaaparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    pcaaparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                    help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    pcaaparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    pcaaparser.add_argument('--outdir',help='output directory,default= same dir as input')

    #*************PCAfilter
    pcafparser = subparser.add_parser('PCAfilter',help='PCA filter')
    pcafparser.set_defaults(which='PCAfilter')
    pcafparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    pcafparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcafparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    pcafparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    pcafparser.add_argument('--modeltargetcol',type=int, default =None,help='Target column #.default = last col')
    pcafparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    pcafparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    pcafparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    pcafparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                    help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    pcafparser.add_argument('--ncomponents',type=int,default=2,help='# of components to keep,default =2')
    pcafparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    pcafparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************scattermatrix
    sctrmparser = subparser.add_parser('scattermatrix',help='Scatter matrix of all predictors and target')
    sctrmparser.set_defaults(which = 'scattermatrix')
    sctrmparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    sctrmparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    sctrmparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    sctrmparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    sctrmparser.add_argument('--modeltargetcol',type=int, default =None,help='Target column #.default = last col')
    sctrmparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    sctrmparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    sctrmparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    sctrmparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                    help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    sctrmparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    sctrmparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************qclin all features cross plotting
    qcparser = subparser.add_parser('qclin',help='Cross plot predictor vs target to check linearity')
    qcparser.set_defaults(which = 'qclin')
    qcparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    qcparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qcparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    qcparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving.default= 0')
    qcparser.add_argument('--modeltargetcol',type=int, default =None,help='Target column #.default = last col')
    qcparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    qcparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    qcparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    qcparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                    help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    qcparser.add_argument('--polydeg',type = int, default =1,
                            help='degree of polynomial to fit data. default = 1, i.e. st line')
    qcparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    qcparser.add_argument('--heatonly',action='store_true',default=False,
                        help='Only plot heat map. default =plot and save all plots')
    qcparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************Linear Regression Model fitting only
    lrparser = subparser.add_parser('linreg',help='Linear Regression Model fit ')
    lrparser.set_defaults(which = 'linreg')
    lrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    lrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    lrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    lrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    lrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    lrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    lrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving.default= 0')

    lrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    lrparser.add_argument('--scaletarget',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')
    lrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    lrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    lrparser.add_argument('--outdir',help='output directory,default= same dir as input')




    #*************featureranking for regression
    frparser = subparser.add_parser('featureranking',help='Ranking of regression attributes')
    frparser.set_defaults(which = 'featureranking')
    frparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    frparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    frparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    frparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    frparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    frparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    frparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    frparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving.default= 0')

    frparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')

    frparser.add_argument('--testfeatures',choices=['rfe','mutualinforeg','svr','svrcv','rfregressor','decisiontree'],default='rfregressor',
                            help='Test for features significance: Randomized Lasso, recursive feature elimination #,default= rfregressor')
    #lassalpha is used with randomized lasso only
    frparser.add_argument('--lassoalpha',type=float,default = 0.025,help='alpha = 0 is OLS. default=0.005')
    #features2keep is used with svr only
    frparser.add_argument('--features2keep',type=int,default=5,help='#of features to keep.default=5')
    #following 2 are used with any cross validation e.g. random forest regressor, svrcv
    frparser.add_argument('--cv',type = int,default = 3,help='Cross Validation nfold. default=3')
    frparser.add_argument('--traintestsplit',type=float,default =.3,help='Train Test split. default = 0.3')





    #*************linfitpredict
    lfpparser = subparser.add_parser('linfitpredict',help='Linear Regression fit on one data set and predicting on another ')
    lfpparser.set_defaults(which='linfitpredict')
    lfpparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    lfpparser.add_argument('predictiondatacsv',help='csv file to predict at')
    lfpparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lfpparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    lfpparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    lfpparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    lfpparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    lfpparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    lfpparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving.default= 0')

    lfpparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lfpparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    lfpparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    lfpparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    lfpparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    lfpparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    lfpparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID column of prediction data to remove before predicting. They added back before saving.')
    lfpparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')
    lfpparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    lfpparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    lfpparser.add_argument('--outdir',help='output directory,default= same dir as input')




    #*************KNNtest
    knntparser = subparser.add_parser('KNNtest',help='Test number of nearest neighbors for KNN')
    knntparser.set_defaults(which='KNNtest')
    knntparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    knntparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    knntparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    knntparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving.')
    knntparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    knntparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    knntparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    knntparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    knntparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    knntparser.add_argument('--cv',type = int,default = 3,help='Cross Validation nfold. default=3')
    knntparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    knntparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************KNNfitpredict
    knnfparser = subparser.add_parser('KNNfitpredict',help='KNN fit on one data set and predicting on another ')
    knnfparser.set_defaults(which='KNNfitpredict')
    knnfparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    knnfparser.add_argument('predictiondatacsv',help='csv file to predict at')
    knnfparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    knnfparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    knnfparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID column of data to model to remove before model fitting. They are added back before saving')
    knnfparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    knnfparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    knnfparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    knnfparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')

    knnfparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    knnfparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    knnfparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    knnfparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    knnfparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    knnfparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    knnfparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving.')
    knnfparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')
    knnfparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')
    knnfparser.add_argument('--kneighbors',type=int,default=10,help='# of nearest neighbors. default = 10')
    knnfparser.add_argument('--outdir',help='output directory,default= same dir as input')
    knnfparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')


    #*************TuneCatBoostRegressor
    tcbrparser = subparser.add_parser('TuneCatBoostRegressor',help='Hyper Parameter Tuning of CatBoost Regression')
    tcbrparser.set_defaults(which = 'TuneCatBoostRegressor')
    tcbrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    tcbrparser.add_argument('predictiondatacsv',help='csv file to predict at')
    tcbrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    tcbrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    tcbrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    tcbrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    tcbrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    tcbrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')
    # lrparser.add_argument('--targetencode',action='store_true',default=False,
                    # help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')

    tcbrparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbrparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    tcbrparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    tcbrparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    tcbrparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    tcbrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    tcbrparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving.')
    tcbrparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')

    tcbrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    tcbrparser.add_argument('--iterations',type=int,nargs='+',default=[10,500,1000,5000],
                        help='Learning Iterations, default =[10,500,1000,5000]')
    tcbrparser.add_argument('--learningrate',type=float,nargs= '+', default=[0.01,0.03,0.1],
                        help='learning_rate. default=[0.01,0.03,0.1]')
    tcbrparser.add_argument('--depth',type=int,nargs = '+',default=[2,4,6,8],help='depth of trees. default=[2,4,6,8]')
    tcbrparser.add_argument('--cv',type=int,default=3,help='Cross Validation default=3')
    tcbrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcbrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')



    #*************CatBoostRegressor
    cbrparser = subparser.add_parser('CatBoostRegressor',help='CatBoost Regressor')
    cbrparser.set_defaults(which='CatBoostRegressor')
    cbrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    cbrparser.add_argument('predictiondatacsv',help='csv file to predict at')
    cbrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cbrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    cbrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    cbrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    cbrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    cbrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    cbrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    cbrparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cbrparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    cbrparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    cbrparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    cbrparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    cbrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    cbrparser.add_argument('--predictionidcol',type=int,default= [0], nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving. default= 0')
    cbrparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')

    cbrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    cbrparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbrparser.add_argument('--learningrate',type=float,default=0.03,help='learning_rate. default=0.03')
    cbrparser.add_argument('--depth',type=int,default=2,help='depth of trees. default=2')
    cbrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbrparser.add_argument('--featureimportance',action='store_true',default=False,
                    help='List feature importance.default= False')
    cbrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    cbrparser.add_argument('--outdir',help='output directory,default= same dir as input')



    #*************NuSVR
    nsvrparser = subparser.add_parser('NuSVR',help='Nu Support Vector Regressor')
    nsvrparser.set_defaults(which='NuSVR')
    nsvrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    nsvrparser.add_argument('predictiondatacsv',help='csv file to predict at')
    nsvrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    nsvrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    nsvrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    nsvrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    nsvrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    nsvrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    nsvrparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvrparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    nsvrparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    nsvrparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    nsvrparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    nsvrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    nsvrparser.add_argument('--predictionidcol',type=int,default= [0], nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving. default= 0')
    nsvrparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')

    nsvrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    nsvrparser.add_argument('--nu',type=float,default=0.5,help='upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. value between 0 1, default =0.5')
    nsvrparser.add_argument('--errpenalty',type=float,default=1.0,help='error penalty. default=1.0')
    nsvrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nsvrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    nsvrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    nsvrparser.add_argument('--outdir',help='output directory,default= same dir as input')







    #*************ANNRegressor
    annrparser = subparser.add_parser('ANNRegressor',help='ANN Regressor')
    annrparser.set_defaults(which='ANNRegressor')
    annrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    annrparser.add_argument('predictiondatacsv',help='csv file to predict at')
    annrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    annrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    annrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    annrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    annrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    annrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    annrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    annrparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    annrparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    annrparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    annrparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    annrparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    annrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    annrparser.add_argument('--predictionidcol',type=int,default=[0,1], nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving.Default = 0 1')
    annrparser.add_argument('--minmaxscale',action='store_true',default=False,
                    help='Apply min max scaler to scale predicted output to input range. default= False i.e. no scaling')

    annrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,help='Min Max scale limits. default=use input data limits ')

    annrparser.add_argument('--nodes',type=int,nargs='+',help='# of nodes in each layer. no defaults.repeat for number of layers')
    annrparser.add_argument('--activation',choices=['relu','sigmoid'],nargs = '+',
                            help='activation per layer.choices: relu or sigmoid. no defaults, repeat for number of layers')
    annrparser.add_argument('--epochs',type=int,default=100,help='depth of trees. default=100')
    annrparser.add_argument('--batch',type=int,default=5,help='depth of trees. default=5')

    annrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    annrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    annrparser.add_argument('--radius',default=5000.00, type=float,
                            help='search radius for map interpolation. dfv=5000m using idw')
    annrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    annrparser.add_argument('--outdir',help='output directory,default= same dir as input')




    #*************testCmodels
    tcmdlparser = subparser.add_parser('testCmodels',help='Test Classification models')
    tcmdlparser.set_defaults(which='testCmodels')
    tcmdlparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    tcmdlparser.add_argument('predictiondatacsv',help='csv file to predict at')
    tcmdlparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcmdlparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    tcmdlparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    tcmdlparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    tcmdlparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    tcmdlparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    tcmdlparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')
    tcmdlparser.add_argument('--qcut',type=int,default=True,help='Divisions to input target data set. default = 3')
    tcmdlparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    tcmdlparser.add_argument('--cv',type = int,default = 3,help='Cross Validation nfold. default=3')
    tcmdlparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcmdlparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    #*************logisticreg
    lgrparser = subparser.add_parser('logisticreg',help='Apply Logistic Regression Classification')
    lgrparser.set_defaults(which = 'logisticreg')
    lgrparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    lgrparser.add_argument('predictiondatacsv',help='csv file to predict at')
    lgrparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    lgrparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    lgrparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    lgrparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    lgrparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. Theyare added back before saving.')

    lgrparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    lgrparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    lgrparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    lgrparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    lgrparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID column of prediction data to remove before predicting. They are added back before saving.')
    lgrparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    lgrparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    lgrparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    lgrparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    lgrparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    lgrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    lgrparser.add_argument('--cv',type=int,help='Cross Validation default=None')
    lgrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    lgrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    #*************GaussianNqiveBayes
    nbparser = subparser.add_parser('GaussianNaiveBayes',help='Apply Gaussian Naive Bayes Classification')
    nbparser.set_defaults(which = 'GaussianNaiveBayes')
    nbparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    nbparser.add_argument('predictiondatacsv',help='csv file to predict at')
    nbparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    nbparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    nbparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    nbparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    nbparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    nbparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    nbparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    nbparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    nbparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    nbparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving.')
    nbparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    nbparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    nbparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    nbparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    nbparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    nbparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nbparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    nbparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nbparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')


    #*************Quadratic Discriminant Analysis
    qdaparser = subparser.add_parser('QDA',help='Apply Quadratic Discriminant Analysis Classification')
    qdaparser.set_defaults(which = 'QDA')
    qdaparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    qdaparser.add_argument('predictiondatacsv',help='csv file to predict at')
    qdaparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    qdaparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    qdaparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    qdaparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    qdaparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    qdaparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    qdaparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    qdaparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    qdaparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    qdaparser.add_argument('--predictionidcol',type=int,default=[0], nargs='+',
                    help='ID columns of prediction data to remove before predicting. They are added back before saving.')
    qdaparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    qdaparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    qdaparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    qdaparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    qdaparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    qdaparser.add_argument('--outdir',help='output directory,default= same dir as input')
    qdaparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    qdaparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    qdaparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')










    #*************TuneCatBoostClassifier
    tcbcparser = subparser.add_parser('TuneCatBoostClassifier',help='Hyper Parameter Tuning of CatBoost Classification - Multi Class')
    tcbcparser.set_defaults(which = 'TuneCatBoostClassifier')
    tcbcparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    tcbcparser.add_argument('predictiondatacsv',help='csv file to predict at')
    tcbcparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbcparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    tcbcparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    tcbcparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    tcbcparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    tcbcparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.')

    tcbcparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbcparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    tcbcparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    tcbcparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    tcbcparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    tcbcparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID column of prediction data to remove before predicting. They are added back before saving.')
    tcbcparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    tcbcparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    tcbcparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    tcbcparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    tcbcparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    tcbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcbcparser.add_argument('--iterations',type=int,nargs='+',default=[10,500,1000,5000],
                        help='Learning Iterations, default =[10,500,1000,5000]')
    tcbcparser.add_argument('--learningrate',type=float,nargs= '+', default=[0.01,0.03,0.1],
                        help='learning_rate. default=[0.01,0.03,0.1]')
    tcbcparser.add_argument('--depth',type=int,nargs = '+',default=[2,4,6,8],help='depth of trees. default=[2,4,6,8]')
    tcbcparser.add_argument('--cv',type=int,default=3,help='Cross Validation default=3')
    tcbcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    tcbcparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    #*************CatBoostClassifier
    cbcparser = subparser.add_parser('CatBoostClassifier',help='Apply CatBoost Classification - Multi Class')
    cbcparser.set_defaults(which = 'CatBoostClassifier')
    cbcparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    cbcparser.add_argument('predictiondatacsv',help='csv file to predict at')
    cbcparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cbcparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    cbcparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    cbcparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    cbcparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    cbcparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    cbcparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')

    cbcparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cbcparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    cbcparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    cbcparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    cbcparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    cbcparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    cbcparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID columns of prediction data to remove before predicting. The are added back before saving.')
    cbcparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    cbcparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    cbcparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    cbcparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    cbcparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    cbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    cbcparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbcparser.add_argument('--learningrate',type=float,default=0.3,help='learning_rate. default=0.3')
    cbcparser.add_argument('--depth',type=int,default=2,help='depth of trees. default=2')
    cbcparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbcparser.add_argument('--featureimportance',action='store_true',default=False,
                        help='List feature importance.default= False')
    cbcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbcparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')


    #*************ClusterTest
    clparser = subparser.add_parser('clustertest',help='Testing of KMeans # of clusters using elbow plot')
    clparser.set_defaults(which='clustertest')
    clparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    clparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    clparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    clparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    clparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    clparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    clparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    clparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    clparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    clparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    clparser.add_argument('--outdir',help='output directory,default= same dir as input')



    #*************clustering
    cl1parser = subparser.add_parser('clustering',help='Apply KMeans clustering')
    cl1parser.set_defaults(which='clustering')
    cl1parser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    cl1parser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cl1parser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    cl1parser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    cl1parser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    cl1parser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    cl1parser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    cl1parser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    cl1parser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    cl1parser.add_argument('--nclusters',type=int,default=5,help='# of clusters. default = 5')
    cl1parser.add_argument('--plotsilhouette',action='store_true',default=False,help='Plot Silhouete. default=False')
    cl1parser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
                            help='add cluster labels or binary dummies.default=labels')
    cl1parser.add_argument('--outdir',help='output directory,default= same dir as input')
    cl1parser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')




    #*************GaussianMixtureModel
    '''
    you can input only the model data file -> results in fitting gaussian to specified # of clusters
    If you enter also seismic attributes then prediction using the defined clusters is done
    '''
    gmmparser = subparser.add_parser('GaussianMixtureModel',help='Apply GaussianMixtureModel Classification')
    gmmparser.set_defaults(which = 'GaussianMixtureModel')
    gmmparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    gmmparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    gmmparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    gmmparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    gmmparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    gmmparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. Theyare added back before saving.')
    gmmparser.add_argument('--modelbayesian',action='store_true',default=False,
                        help='Bayesian Gauusian Mixture Model. default= use Gaussian Mixture Model')
    gmmparser.add_argument('--modelncomponents',type=int,default=4,help='# of clusters.default=4')

    gmmparser.add_argument('--predictiondatacsv',help='csv file to predict at')
    gmmparser.add_argument('--predictioncolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--predictioncolselect',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--sampleprediction',type=float,default=1.0,help='fraction of prediction data.default=1')
    gmmparser.add_argument('--predictionscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    gmmparser.add_argument('--predictionscalesave',action='store_true',default=False,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    gmmparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    gmmparser.add_argument('--predictionidcol',type=int, nargs='+',
                    help='ID column of prediction data to remove before predicting. They are added back before saving.')
    gmmparser.add_argument('--classweight',action='store_true',default=False,
                        help='Balance classes by proportional weighting. default =False -> no balancing')
    gmmparser.add_argument('--coded',action='store_true',default=False,
                        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    gmmparser.add_argument('--targetencode',action='store_true',default=False,
                    help='Set to True for Classifiction i.e. label encoding. default=False, i.e. regression')
    gmmparser.add_argument('--qcut',type=int,default=True,help='Divisions to model input target data set. default = True')
    gmmparser.add_argument('--nqcutclasses',type=int,default=3,help='# of qcut classes. default= 3')
    gmmparser.add_argument('--outdir',help='output directory,default= same dir as input')
    gmmparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')




    #*************tSNE
    tsneparser = subparser.add_parser('tSNE',help='Apply tSNE (t distribution Stochastic Neighbor Embedding) clustering')
    tsneparser.set_defaults(which='tSNE')
    tsneparser.add_argument('allattribcsv',help='csv file will all attributes')
    tsneparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],\
    help='Columns to use for clustering. default= 3 4 5 ')
    tsneparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tsneparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    tsneparser.add_argument('--targetcol',type=int,default = None,
                    help='Target column # to add back. You do not have to add a target default = none')
    tsneparser.add_argument('--learningrate',type=int,default=200,help='Learning rate. default=200')
    tsneparser.add_argument('--sample',type=float,default=.2,help='fraction of data of sample.default=0.2')
    tsneparser.add_argument('--outdir',help='output directory,default= same dir as input')


    #*************DBSCAN
    dbsnparser = subparser.add_parser('DBSCAN',help='Apply DBSCAN (Density Based Spatial Aanalysis with Noise) clustering')
    dbsnparser.set_defaults(which='DBSCAN')
    dbsnparser.add_argument('modeldatacsv',help='csv file of all attributes at well locations to fit model')
    dbsnparser.add_argument('--modelcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    dbsnparser.add_argument('--modelcolselect',type=int,nargs='+',help='Predictor column #s, no default')
    dbsnparser.add_argument('--samplemodel',type=float,default=1.0,help='fraction of model data.default=1')
    dbsnparser.add_argument('--modeltargetcol',type=int, default =-1,help='Target column #.default = last col')
    dbsnparser.add_argument('--modelscalefeatures',action='store_false',default=True,
                    help='Do not scale features. default=true, i.e. scale features')
    dbsnparser.add_argument('--modelscalesave',action='store_false',default=True,
                    help='Use to apply previously saved scaler to prediction or validation or test data.default=True = compute scaler, apply it and save it to file.')
    dbsnparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    dbsnparser.add_argument('--modelidcol',type=int, nargs='+',
                    help='ID columns of data to model to remove before model fitting. They are added back before saving.default= 0')
    dbsnparser.add_argument('--eps',type=float,default=0.5,help='eps. default=0.5')
    dbsnparser.add_argument('--minsamples',type=int,default=10,help='minsamples. default=10')
    dbsnparser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
                            help='add cluster labels or binary dummies.default=labels')
    dbsnparser.add_argument('--outdir',help='output directory,default= same dir as input')



    #*************semisupervised
    sspparser = subparser.add_parser('semisupervised',help='Apply semi supervised Class prediction ')
    sspparser.set_defaults(which = 'semisupervised')
    sspparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    sspparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    sspparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    sspparser.add_argument('--wtargetcol',type=int,default= -1,help='Target column # in well csv file. default = last column')
    sspparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    sspparser.add_argument('--col2drop',type=int,default=None,help='drop column in case of scaled target.default=None')
    #sspparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    sspparser.add_argument('--sample',type=float,default=.005,help='fraction of data of sample.default=0.005')
    sspparser.add_argument('--outdir',help='output directory,default= same dir as input')
    sspparser.add_argument('--nneighbors',type=int ,default=7,help='Used with knn to classify data.default=7')
    sspparser.add_argument('--kernel',choices=['knn','rbf'],default='knn',
                        help='Kernel for semi supervised classification.default= knn')




    if not oneline:
        result = mainparser.parse_args()
    else:
        result = mainparser.parse_args(oneline)

    #result = mainparser.parse_args()
    if result.which not in allcommands:
        mainparser.print_help()
        exit()
    else:
        return result



def main():
    sns.set()
    def process_commands():
        print(cmdl.which)


        if cmdl.which =='dropcols':
            process_dropcols(cmdl.csvfile,
                            cmdlcols2drop=cmdl.cols2drop,
                            cmdloutdir=cmdl.outdir)

        elif cmdl.which =='listcsvcols' :
            process_listcsvcols(cmdl.csvfile)


        elif cmdl.which =='featurescale':
            process_featurescale(cmdl.csvfile,
                            cmdltargetcol=cmdl.targetcol,
                            cmdlidcol=cmdl.idcol,
                            cmdltargetencode = cmdl.targetencode,
                            cmdlscalefeatures=cmdl.scalefeatures,
                            cmdlkind=cmdl.kind,cmdlscalesave=cmdl.scalesave,
                            cmdldatacolsrange = cmdl.datacolsrange,
                            cmdldatacolselect = cmdl.datacolselect,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdloutdir=cmdl.outdir)


        elif cmdl.which =='PCAanalysis':
            process_PCAanalysis(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlkind = cmdl.kind,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)



        elif cmdl.which =='PCAfilter':
            process_PCAfilter(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlkind = cmdl.kind,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlncomponents = cmdl.ncomponents,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)



        elif cmdl.which =='scattermatrix':
            process_scattermatrix(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)

        elif cmdl.which =='featureranking':
            process_feature_ranking(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdltestfeatures = cmdl.testfeatures,
                            cmdlcv = cmdl.cv,
                            cmdlfeatures2keep = cmdl.features2keep,
                            cmdllassoalpha = cmdl.lassoalpha)

        elif cmdl.which =='qclin':
            process_qclin(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlheatonly = cmdl.heatonly,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)


        elif cmdl.which =='linreg':
            process_linreg(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlscaletarget = cmdl.scaletarget,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)





        elif cmdl.which == 'linfitpredict':
            process_linfitpredict(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)





        elif cmdl.which == 'KNNtest':
            process_KNNtest(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlkind = cmdl.kind,
                            cmdlcv = cmdl.cv,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'KNNfitpredict':
            process_KNNfitpredict(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdlkneighbors=cmdl.kneighbors,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'TuneCatBoostRegressor':
            process_TuneCatBoostRegressor(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdliterations = cmdl.iterations,
                            cmdllearningrate=cmdl.learningrate,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdldepth = cmdl.depth,
                            cmdlcv =cmdl.cv,
                            cmdlfeatureimportance = cmdl.featureimportance,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdloutdir=cmdl.outdir)



        elif cmdl.which == 'CatBoostRegressor':
            process_CatBoostRegressor(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdliterations = cmdl.iterations,
                            cmdllearningrate=cmdl.learningrate,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdldepth = cmdl.depth,
                            cmdlcv =cmdl.cv,
                            cmdlfeatureimportance = cmdl.featureimportance,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdloutdir=cmdl.outdir)



        elif cmdl.which == 'NuSVR':
            process_NuSVR(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdlerrpenalty=cmdl.errpenalty,
                            cmdlnu = cmdl.nu,
                            cmdlscaleminmaxvalues =cmdl.scaleminmaxvalues,
                            cmdlcv =cmdl.cv,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdloutdir=cmdl.outdir)





        elif cmdl.which == 'ANNRegressor':
            process_ANNRegressor(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlminmaxscale=cmdl.minmaxscale,
                            cmdlnodes=cmdl.nodes,
                            cmdlactivation=cmdl.activation,
                            cmdlepochs = cmdl.epochs,
                            cmdlbatch = cmdl.batch,
                            cmdlcv =cmdl.cv,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdloutdir=cmdl.outdir,
                            cmdlradius= cmdl.radius,
                            )



        elif cmdl.which == 'CatBoostClassifier':
            process_CatBoostClassifier(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlfeatureimportance = cmdl.featureimportance,
                            cmdlcv = cmdl.cv,
                            cmdliterations = cmdl.iterations,
                            cmdllearningrate=cmdl.learningrate,
                            cmdldepth = cmdl.depth,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)



        elif cmdl.which == 'TuneCatBoostClassifier':
            process_TuneCatBoostClassifier(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlcv = cmdl.cv,
                            cmdliterations = cmdl.iterations,
                            cmdllearningrate=cmdl.learningrate,
                            cmdldepth = cmdl.depth,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)




        elif cmdl.which == 'testCmodels':
            process_testCmodels(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlcv = cmdl.cv)




        elif cmdl.which == 'logisticreg':
            process_logisticreg(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlcv  = cmdl.cv,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)



        elif cmdl.which == 'GaussianNaiveBayes':
            process_GaussianNaiveBayes(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlcv = cmdl.cv,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)




        elif cmdl.which == 'QDA':
            process_QuadraticDiscriminantAnalysis(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlcv = cmdl.cv,
                            cmdlvalsize=cmdl.valsize,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)









        elif cmdl.which =='clustertest':
            process_clustertest(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)


        elif cmdl.which == 'clustering':
            process_clustering(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlnclusters=cmdl.nclusters,
                            cmdlplotsilhouette=cmdl.plotsilhouette,
                            cmdladdclass=cmdl.addclass,
                            cmdloutdir=cmdl.outdir,
                            cmdlhideplot=cmdl.hideplot)





        elif cmdl.which =='DBSCAN':
            process_dbscan(cmdl.modeldatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlkind = cmdl.kind,
                            cmdladdclass=cmdl.addclass,
                            cmdleps=cmdl.eps,
                            cmdlminsamples=cmdl.minsamples,
                            cmdloutdir=cmdl.outdir)


        # GMM clustering / prediction
        elif cmdl.which == 'GaussianMixtureModel':
            process_GaussianMixtureModel(cmdl.modeldatacsv,cmdl.predictiondatacsv,
                            cmdlmodelcolsrange=cmdl.modelcolsrange,
                            cmdlmodelcolselect = cmdl.modelcolselect,
                            cmdlmodeltargetcol=cmdl.modeltargetcol,
                            cmdlmodelidcol = cmdl.modelidcol,
                            cmdlsamplemodel=cmdl.samplemodel,
                            cmdlmodelscalefeatures = cmdl.modelscalefeatures,
                            cmdlmodelscalesave = cmdl.modelscalesave,
                            cmdlpredictioncolsrange=cmdl.predictioncolsrange,
                            cmdlpredictioncolselect = cmdl.predictioncolselect,
                            cmdlpredictionscalefeatures = cmdl.predictionscalefeatures,
                            cmdlsampleprediction=cmdl.sampleprediction,
                            cmdlpredictionscalesave = cmdl.predictionscalesave,
                            cmdlkind = cmdl.kind,
                            cmdlpredictionidcol = cmdl.predictionidcol,
                            cmdlqcut=cmdl.qcut,
                            cmdlnqcutclasses = cmdl.nqcutclasses,
                            cmdltargetencode=cmdl.targetencode,
                            cmdlcoded = cmdl.coded,
                            cmdloutdir=cmdl.outdir,
                            cmdlmodelbayesian  = cmdl.modelbayesian,
                            cmdlmodelncomponents=cmdl.modelncomponents,
                            cmdlhideplot=cmdl.hideplot,
                            cmdlclassweight=cmdl.classweight)






        elif cmdl.which == 'tSNE':
            swa = pd.read_csv(cmdl.allattribcsv)
            swaxx = swa.sample(frac=cmdl.sample).copy()
            if cmdl.colsrange:
                print('Attrib From col# %d to col %d' %(cmdl.colsrange[0],cmdl.colsrange[1]))
                swax = swaxx[swaxx.columns[cmdl.colsrange[0]: cmdl.colsrange[1]]]
            else:
                swax = swa[swaxx.columns[cmdl.cols2cluster]]

            xyzc = swaxx[swaxx.columns[cmdl.xyzcols]]

            clustering = TSNE(learning_rate = cmdl.learningrate)
            tsne_features = clustering.fit_transform(swaxx)


            xs =tsne_features[:,0]
            ys = tsne_features[:,1]
            plt.scatter(xs,ys,alpha=0.5)
            plt.show()


            swaxx['tSNE0'] = tsne_features[:,0]
            swaxx['tSNE1'] = tsne_features[:,1]

            savefiles(seisf = cmdl.allattribcsv,
                        sdf = swaxx, sxydf = swaxx,
                        outdir = cmdl.outdir,
                        ssuffix ='_tsne')


        elif cmdl.which == 'semisupervised':
            process_semisupervised(cmdl.wellattribcsv,cmdl.seisattribcsv,
                    cmdlwtargetcol=cmdl.wtargetcol,cmdlwcolsrange=cmdl.wcolsrange,
                    cmdlwellsxyzcols = cmdl.wellsxyzcols,
                    cmdlsample=cmdl.sample,cmdlnneighbors = cmdl.nneighbors,
                    cmdlkernel=cmdl.kernel,cmdlcol2drop= cmdl.col2drop,
                    cmdloutdir=cmdl.outdir)

    #print(__doc__)
    cmdl = getcommandline()
    if cmdl.which == 'workflow':
        lnum =0
        startline = cmdl.startline
        with open(cmdl.commandfile,'r') as cmdlfile:
            for line in cmdlfile:
                lnum +=1
                print()
                print('%00d:>'% lnum,line)
                if lnum >= startline:
                    parsedline =shlex.split(line)[2:]
                    #if len(parsedline) >=1:
                    if len(parsedline) >=1 and not cmnt(line):
                        cmdl = getcommandline(*parsedline)
                        process_commands()
                else:
                    print('Skip line:%00d' % lnum,line)
    else:
        process_commands()








if __name__=='__main__':
	main()

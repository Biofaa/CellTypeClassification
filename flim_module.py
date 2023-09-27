import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage.morphology as morph
import skimage.filters as filters
import skimage.measure as measure
from skimage.morphology import opening
from skimage.morphology import h_maxima
from skimage.measure import moments
import pandas as pd
import struct
import zlib
import tkinter
from tkinter import filedialog

class decode:
    def getfile(filetype='*'):
        '''Ask single or multiple files and returns a list of filename+filepath'''
        root = tkinter.Tk()
        root.withdraw() #hides tkinter window
        # currdir = os.getcwd()
        if type(filetype)==str:
            filetype=[filetype]
        for i in range(0, np.size(filetype, axis=0)):
            filetype[i]=(filetype[i]+' files','*.'+filetype[i])
        filename = filedialog.askopenfilenames(filetypes=filetype)
        if not type(filename)=='NoneType':
            return filename

    def savefigure(filename='plot', save=1, transparent=True, extension='png'):
        if filename!='plot':
           plt.savefig(filename+'.'+extension, dpi=500, bbox_inches='tight', transparent=transparent) 
           return
        if save==1:
            print('select path where to save figure')
            filepath=tkinter.filedialog.asksaveasfilename(filetypes=(
                ("PNG files", "*.png"),("SVG files", "*.svg")))
            plt.savefig(filepath+'.'+extension, dpi=500, bbox_inches="tight", transparent=transparent)
        elif save==-1:
            plt.savefig('C:/Users/Fabio/Desktop/'+filename+'.'+extension, dpi=500, bbox_inches='tight', transparent=transparent)
        
    def asksavefile(filetype='*'):
        print('select path where to save file')
        if type(filetype)==str:
            filetype=[filetype]
        for i in range(0, np.size(filetype, axis=0)):
            filetype[i]=(filetype[i]+' files','*.'+filetype[i])
        filename=tkinter.filedialog.asksaveasfilename(filetypes=filetype)
        if filetype!='*':
            if filename.split('/')[-1].find('.')!=-1:
                filename=filename.split('.')[0:-1]
            filename=filename+filetype[0][1][1:]
        return filename
        
    def getpath():
        '''Ask single or multiplefiles and returns filepath'''
        root = tkinter.Tk()
        root.withdraw() #hides tkinter window
        # currdir = os.getcwd()
        filename = filedialog.askdirectory()
        if not type(filename)=='NoneType':
            return filename
        
    def get_file_extension():
        filename=decode.getfile()[0]
        # filename=filename[0]
        file_extension=filename.split('.')[-1]
        return file_extension

    # def getfilename():
    #     '''Ask single or multiple .R64 and .ref files and returns a list of filename+filepath'''
    #     root = tkinter.Tk()
    #     root.withdraw() #hides tkinter window
    #     # currdir = os.getcwd()
    #     filename = filedialog.askopenfilenames(filetypes =[('R64 files', '*.R64'),('ref files', '*.ref')])
    #     if not type(filename)=='NoneType':
    #         return filename

    #function to determine R64 size
    def sizeR64(R64ref):
        #Im_size is a tuple, one must extract the int32 value at position [0]
    	Im_size = struct.unpack('<i', zlib.decompressobj().decompress(R64ref, 4))[0]
    	return Im_size

    #function to decode R64
    def decodeR64(R64ref):
    	Im_size=decode.sizeR64(R64ref[0:256])
    	cell_size=5*Im_size*Im_size
    	dp=np.dtype('<f4')
    	bufsize = cell_size * dp.itemsize + 4
    	data = zlib.decompress(R64ref, bufsize=bufsize)
    	data = np.frombuffer(data, '<' + dp.char, offset=4)
    	data = data[: cell_size].copy()
    	data = np.reshape(data, (-1, Im_size, Im_size))
    	return data

    def import_R64(har2=False, filename=-1):
        '''
        g,s,intensity, filename=importfile()
        Extracts data from .R64 files
        '''
        intensity=[]
        har1_phase=[] #classic FLIM phase
        har1_modulation=[] #classifc FLIM modulation
        har2_phase=[]
        har2_modulation=[]
        if filename==-1:
            print('choose a R64 file to open:')
            filename=decode.getfile('R64')
        elif type(filename)==str:
                filename=[filename]
        namefile=[]
        for i in range(0, np.size(filename)):
            with open(filename[i], 'rb') as file: # b is important -> binary
                fileContent = file.read()
            data = decode.decodeR64(fileContent)
            #1st harmonic data for each pixel
            intensity.append(data[0,:,:])
            har1_phase.append(data[1,:,:])
            har1_modulation.append(data[2,:,:])
            har2_phase.append(data[3,:,:])
            har2_modulation.append(data[4,:,:])
            # get filenames
            namefile.append(filename[i].split('/')[-1])
        intensity=np.array(intensity, dtype='float64')
        har1_modulation=np.array(har1_modulation, dtype='float64')
        har1_phase=np.array(har1_phase, dtype='float64')
        g,s=gs_conversion(har1_modulation, har1_phase)
        if har2:
            g_har2, s_har2 = gs_conversion(har2_modulation, har2_phase)
            return g,s,intensity, namefile, g_har2, s_har2
        else:
            return g,s,intensity, namefile
        
class metabolism:    
    def gs_to_oxphos(g,s, BrillianceCorrection=False):
        '''
        convert (g,s) point into oxphos/glycolysis
        d1: distance from tau=0.4 ns and phasor point
        d2: distance from tau=3.4 ns and phasor point
        '''
        # get (g,s) of metabolic axis
        g_tau, s_tau=tau_to_gs()
        # calculate distance of a generic point from extrema of metabolic axis
        d1=np.sqrt(np.power(np.subtract(g,g_tau[0]),2)+np.power(np.subtract(s_tau[0],s),2))
        d2=np.sqrt(np.power(np.subtract(g_tau[1],g),2)+np.power(np.subtract(s,s_tau[1]),2))
        # correct for brightness
        if BrillianceCorrection:
            d1=d1/(d1+8.5*d2)
            d2=(8.5*d2)/(d1+8.5*d2)
        oxphos_relative=np.divide(d1, d1+d2)     
        return oxphos_relative
    
    def MetabolicMap(g,s, intensity, namefile, threshold_low=15, threshold_high=-1, extension='svg'):
        # intensity, threshold=phasor.LipofuscinCorrection(intensity, threshold_low)
        g=IntensityCorrection_single(g, intensity, threshold_low, threshold_high)
        s=IntensityCorrection_single(s, intensity, threshold_low, threshold_high)
        cmin=10
        cmax=0
        oxphos=[]
        for i in range (0, np.size(g, axis=0)):
            g_oxphos=g[i,:,:]
            # project phasor points on metabolic axis (line passing through t1=0.4 ns and t2=3.4 ns)
            s_oxphos=np.where(g_oxphos>0, 0.55-0.45*g_oxphos, 0) 
            oxphos.append(np.where(np.logical_and(g_oxphos>0, s_oxphos>0),
                metabolism.gs_to_oxphos(g_oxphos, s_oxphos, BrillianceCorrection=False), 0))
            oxphos[i]=np.where(intensity[i,:,:]<threshold_low, 0, oxphos[i])
            oxphos[i]=np.around(oxphos[i], 2)
            oxphos_tmp=oxphos[i]
            q1, q2, q3 = np.quantile(oxphos_tmp[oxphos_tmp>0], [0.25, 0.5, 0.75])
            if q1<cmin:
                cmin=q1
            if q3>cmax:
                cmax=q3
        oxphos=np.array(oxphos)
        for i in range (0, np.size(g, axis=0)):
            plt.imshow(oxphos[i,:,:], cmap='jet', vmin=cmin, vmax=cmax,
                        alpha=np.float32(np.where(intensity[i,:,:]<threshold_low, 0, 1)))
            plt.title(namefile[i])
            plt.axis('off')
            plt.colorbar()
            plt.savefig('C:/Users/Fabio/Desktop/'
                        +namefile[i].split('.')[0]+'.'+extension, 
                        bbox_inches='tight')
            plt.show()
            
class image:
    def area(image):
        '''
        computes circularity from a binary image
        '''
        image=np.int0(image>0)
        area = np.count_nonzero(image)
        return area    
    
    def perimeter(image):
        '''
        computes perimeter from a binary image
        '''
        image=np.int0(image>0)
        perimeter = measure.perimeter(image)
        return perimeter
    
    def circularity(image):
        '''
        computes circularity from a binary image
        '''
        image=np.int0(image>0)
        perimeter = measure.perimeter(image)
        area = np.count_nonzero(image)
        circularity=(4*np.pi*area)/(perimeter**2)
        return circularity
        

def tau_to_gs(tau=[0.4, 3.4], freq=80):
    '''
    perform conversion of tau in (g,s) coordinates
    tau: fluorophore time constant [ns]
    freq: laser repetition rate [MHz]
    '''
    tau=np.multiply(tau,np.float_power(10, -9))
    omega=np.multiply(2,np.multiply(np.pi,freq*np.power(10,6)))
    x=np.multiply(tau,omega)
    # formula for 
    g=np.divide(1,(1+np.square(x)))
    s=np.divide(x,(1+np.square(x)))
    return g,s

def gs_to_tau(g,s, freq=80):
    '''
    tau_m, tau_phi = gs_to_tau(g,s, freq=80)
    perform (g,s) coordinates in tau of modulation and phase
    tau: fluorophore time constant [ns]
    freq: laser repetition rate [MHz]
    '''
    m=np.sqrt(np.square(g)+np.square(s))
    omega=np.multiply(2,np.multiply(np.pi,freq*np.power(10,6)))
    tau_p=(1/omega)*(s/g)
    tau_m=(1/omega)*np.sqrt(1/np.square(m)-1)
    return tau_m,tau_p

def phasor(ph,md):
  #valori singoli pixel
	ph1=np.concatenate(np.radians(ph))
	md1=np.concatenate(md)
	universal_x=md1*np.cos(ph1)
	universal_y=md1*np.sin(ph1)
	return universal_x, universal_y

def universal_circle():
	#semicerchio universale
	theta = np.linspace(0, np.pi, 50)
	r =0.5
	C_x = 0.5+r*np.cos(theta)
	C_y = r*np.sin(theta)
	universal_circle=plt.plot(C_x,C_y,linewidth=1,color='k')
	return universal_circle

#function to enforce a median_filter
def median_filter(g,s, kernel_size=3):
    # dim3=np.size(g, axis=0)
    # for i in range(0, dim3):
    #     g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
    #     hist, g_edges, s_edges=np.histogram2d(g_clean, s_clean, bins=100)
    #     hist=ndimage.median_filter(hist, size=kernel_size)
    #     dim=np.shape(hist)
    #     for j in range(0, dim[0]):
    #         for k in range(0, dim[1]):
    #             condition_1=hist[j][k]>0
    #             condition_2=np.logical_or(np.less_equal(g_clean, g_edges[j]), np.greater_equal(g_clean, g_edges[j+1]))
    #             condition_3=np.logical_or(np.less_equal(s_clean, s_edges[k]), np.greater_equal(s_clean, s_edges[k+1]))
    #             condition=[np.logical_and(condition_1, np.logical_and(condition_2, condition_3))]
    #             g_filtered=g_clean[condition]
    #             s_filtered=s_clean[condition]
    #             # update original matrices
    #     g_new=np.where(g[i,:,:]==g_filtered, g_filtered, 0)
    #     s_new=np.where(s[i,:,:]==s_filtered, s_filtered, 0)
    #     g[i,:,:]=g_new
    #     s[i,:,:]=s_new
    g_filtered=ndimage.median_filter(g, size=kernel_size)
    s_filtered=ndimage.median_filter(s, size=kernel_size)
    return g_filtered, s_filtered

def gaussian_filter(g,s,kernel_size):
    g_filtered=ndimage.median_filter(g, size=kernel_size)
    s_filtered=ndimage.median_filter(s, size=kernel_size)
    return g_filtered, s_filtered


#function for phasor scatter plot 

def scatter(x,y):
    #xedges and yedges based on input dataset
    hist, xedges, yedges = np.histogram2d(x, y, bins=100)
    xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1] - 1)
    c=hist[xidx, yidx]
    histplot=plt.scatter(x, y, c=c, cmap='jet', s=0.2) #takes too much time to plot
    return histplot, c

def contour(x, y):
    x,y=clean(x,y)
    #xedges and yedges based on input dataset
    hist, xedges, yedges = np.histogram2d(x, y, bins=100)
    xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1] - 1)
    # c=hist[xidx, yidx]
    plt.contour(hist, levels=0)
    # plt.contour(hist,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],linewidths=0.5)	
    
def phasorplot(g,s, show=True):
    dim3=np.size(g, axis=0)
    fig=[]
    for i in range(0, dim3):    
        plt.figure()
        g_clean, s_clean=clean(g[i,:,:], s[i,:,:])
        universal_circle()
        fig.append(scatter(g_clean, s_clean)[0])
        fig[i].axes.set_ylim(0,1)
        fig[i].axes.set_box_aspect(1)
        plt.xlabel('g')
        plt.ylabel('s')
        if not show:
            plt.close()
    return fig

def phasorplot_2(g,s, show=True):
    fig=[]  
    plt.figure()
    universal_circle()
    scatter(g, s)
    # fig.axes.set_ylim(0,1)
    # fig.axes.set_box_aspect(1)
    plt.xlabel('g')
    plt.ylabel('s')
    if not show:
        plt.close()
    return fig

def LipofuscinThresholdCalculus(intensity, threshold_low=30):
    '''
    intensity_corrected, threshold_high = LipofuscinCorrection(intensity, threshold_low=30)
    Sets all the pixels out of lower and upper threshold to 0.
    threshold_low: default is zero. Recommended value set the number of frames of the FLIM image.
    '''
    dim3=np.size(intensity, axis=0)
    intensity_correct=[]
    threshold_high=[]
    for i in range(0, dim3):
        intensity_temp=intensity[i,:,:]
        intensity_temp=np.where(intensity_temp>threshold_low, intensity_temp, 0)
        q=np.quantile(np.reshape(intensity_temp[intensity_temp>0], np.size(intensity_temp[intensity_temp>0])), [0.25, 0.75])
        threshold_temp=q[1]+(q[1]-q[0])*1.5
        intensity_temp=np.where(intensity_temp<threshold_temp, intensity_temp, 0)
        # intensity_temp=np.reshape(intensity_temp, (1, np.shape(intensity_temp)[0], np.shape(intensity_temp)[1]))
        intensity_correct.append(intensity_temp)
        threshold_high.append(threshold_temp)
    intensity_correct=np.array(intensity_correct)
    return threshold_high
    

def LipofuscinCorrection(intensity, threshold_low=30):
    '''
    intensity_corrected, threshold_high = LipofuscinCorrection(intensity, threshold_low=30)
    Sets all the pixels out of lower and upper threshold to 0.
    threshold_low: default is zero. Recommended value set the number of frames of the FLIM image.
    '''
    dim3=np.size(intensity, axis=0)
    intensity_correct=[]
    threshold=[]
    for i in range(0, dim3):
        intensity_temp=intensity[i,:,:]
        intensity_temp=np.where(intensity_temp>threshold_low, intensity_temp, 0)
        q=np.quantile(np.reshape(intensity_temp[intensity_temp>0], np.size(intensity_temp[intensity_temp>0])), [0.25, 0.75])
        threshold_temp=q[1]+(q[1]-q[0])*1.5
        intensity_temp=np.where(intensity_temp<threshold_temp, intensity_temp, 0)
        # intensity_temp=np.reshape(intensity_temp, (1, np.shape(intensity_temp)[0], np.shape(intensity_temp)[1]))
        intensity_correct.append(intensity_temp)
        threshold.append(threshold_temp)
    intensity_correct=np.array(intensity_correct)
    return intensity_correct, threshold

def imshow(intensity, cmap='hot', vmin=0, vmax=-1):
    from PIL import Image
    dim3=np.size(intensity, axis=0)
    if dim3>1:
        print('you cannot set vmin and vmax for multiple images passed')
        return
    for i in range(0, dim3):
        if vmax==-1: 
            vmax=np.max(intensity)
        # plt.figure(figsize=(10,10))
        # plt.imshow(intensity[i,:,:], cmap=cmap, interpolation='none', 
                    # vmin=vmin, vmax=vmax)
        # px.imshow(intensity[i,:,:], color_continuous_scale=cmap, zmin=vmin, zmax=vmax)             
        im=Image.fromarray(intensity[i,:,:])
        im.show()
        # plt.axis('off')
        # plt.colorbar(location='left')
        

# def phasorplot_2(g,s):
#     dim3=np.size(g, axis=0)
#     ax=[]
#     for i in range(0, dim3):    
#         figure=plt.figure(figsize=(6,6))
#         axis=figure.add_subplot(projection='scatter_density', aspect=1.0)
#         # axis.set_box_aspect(1)
#         # axis.set_ylim(0,1)
#         # axis.set_xlim(0,1)
#         g_clean, s_clean=clean(g[i,:,:], s[i,:,:])
#         universal_circle()
#         hist, xedges, yedges = np.histogram2d(g_clean, s_clean, bins=100)
#         xidx = np.clip(np.digitize(g_clean, xedges), 0, hist.shape[0] - 1)
#         yidx = np.clip(np.digitize(s_clean, yedges), 0, hist.shape[1] - 1)
#         c=hist[xidx, yidx]
#         axis.scatter_density(g_clean, s_clean, c=c, cmap='jet', aspect='equal')
#         ax.append(axis)
#         # fig.append(scatter(g_clean, s_clean)[0])
#     return ax

def gs_conversion(modulation, phase):
    g=modulation*np.cos(np.radians(phase))
    s=modulation*np.sin(np.radians(phase))
    return g, s

def purgenans(g, s=np.nan):
    '''
    purge all NaNs and nonfinite values in g AND s and returns 1-D vectors
    g: vector or matrix
    '''
    if np.all(np.isnan(s))==True:
        g_clean=g[np.isfinite(g)]
        return g_clean
    else:
        g_clean=g[np.logical_and(np.isfinite(g), np.isfinite(s))]
        s_clean=s[np.logical_and(np.isfinite(g), np.isfinite(s))]
        return g_clean,s_clean

def IntensityCorrection_single(x, intensity, threshold_low=0, threshold_high=-1):
    if np.size(threshold_high)==1:
    # if no threshold are set, does not filter
        if threshold_high==-1:
            threshold_high=np.max(intensity)
        x=np.where(np.logical_and(intensity>threshold_low, intensity<threshold_high), x, 0)
    else:
        for i in range(0, np.size(intensity, axis=0)):
            # calculate
            x[i,:,:]=np.where(np.logical_and(intensity[i,:,:]>threshold_low, intensity[i,:,:]<threshold_high[i]), x[i,:,:], 0)
    return x

def NumberOfPoints(g, s):
    dim3=np.size(g, axis=0)
    N=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        # number of elements is number of elements in g_clean=s_clean 
        # (all nonzero neither NaN)
        N.append(np.size(g_clean))
    N=np.array(N)
    return N
    

def IntensityThresholdCorrection (g, s, intensity, threshold_low=0, threshold_high=-1):
    g_corrected = IntensityCorrection_single(g, intensity, threshold_low, threshold_high)
    s_corrected = IntensityCorrection_single(s, intensity, threshold_low, threshold_high)
    return g_corrected, s_corrected

def FindCenter(g,s):
    dim3=np.size(g, axis=0)
    g_center=np.array([])
    s_center=np.array([])
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        hist, g_edges, s_edges=histogram(g_clean,s_clean, bins=100)
        # reshape hist because histogram function returns 3D array
        hist=hist[0,:,:]
        hist=np.int0(hist>0)
        kernel=np.ones((3,3))
        hist=opening(hist, kernel)
        # calculate moments and apply definition to find centroid
        M=moments(hist, 1)
        if np.all(np.isfinite(M)) and not np.all(M==0):
            g_center=np.append(g_center, g_edges[0][int(M[1, 0]/M[0, 0])])
            s_center=np.append(s_center, s_edges[0][int(M[0, 1]/M[0, 0])])
        else:
            g_center=np.append(g_center, np.nan)
            s_center=np.append(s_center, np.nan)
    return g_center, s_center

def circularity(g,s):
    dim3=np.size(g, axis=0)
    circularity=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        hist, g_edges, s_edges=histogram(g_clean,s_clean, bins=100)
        # reshape hist because histogram function returns 3D array
        hist=hist[0,:,:]
        hist=np.int0(hist>0)
        kernel=np.ones((3,3))
        hist=opening(hist, kernel)
        # calculate moments and apply definition to find centroid
        perimeter = measure.perimeter(hist)
        area = np.count_nonzero(hist)
        circularity.append((4*np.pi*area)/(perimeter**2))
    circularity=np.array(circularity)
    return circularity

def area(g,s):
    dim3=np.size(g, axis=0)
    area=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        hist, g_edges, s_edges=histogram(g_clean,s_clean, bins=100)
        # reshape hist because histogram function returns 3D array
        hist=hist[0,:,:]
        hist=np.int0(hist>0)
        kernel=np.ones((3,3))
        hist=opening(hist, kernel)
        # calculate moments and apply definition to find centroid
        area.append(np.count_nonzero(hist))
    area=np.array(area)
    return area

def perimeter(g,s):
    dim3=np.size(g, axis=0)
    perimeter=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        hist, g_edges, s_edges=histogram(g_clean,s_clean, bins=100)
        # reshape hist because histogram function returns 3D array
        hist=hist[0,:,:]
        hist=np.int0(hist>0)
        kernel=np.ones((3,3))
        hist=opening(hist, kernel)
        perimeter.append(measure.perimeter(hist))
    perimeter=np.array(perimeter)
    return perimeter

def barycenter(g,s):
    dim3=np.size(g, axis=0)
    g_baryc=[]
    s_baryc=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:], s[i,:,:])
        g_baryc.append(np.mean(g_clean))
        s_baryc.append(np.mean(s_clean))
    g_baryc=np.array(g_baryc)
    s_baryc=np.array(s_baryc)
    return g_baryc, s_baryc

def barycenter_std(g,s):
    dim3=np.size(g, axis=0)
    g_baryc_std=np.array([])
    s_baryc_std=np.array([])
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:], s[i,:,:])
        g_baryc_std=np.append(g_baryc_std, np.std(g_clean))
        s_baryc_std=np.append(s_baryc_std, np.std(s_clean))
    return g_baryc_std, s_baryc_std

def gs_percentile(g,s,percent=[25,50,75]):
    dim3=np.size(g, axis=0)
    # initialize output variables
    g_percentile={}
    s_percentile={}
    for j in percent:
        g_percentile['g_'+str(j)]=[]
        s_percentile['s_'+str(j)]=[]
        
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:], s[i,:,:])
        for j in percent:
            g_percentile['g_'+str(j)].append(np.percentile(g_clean, j))
            s_percentile['s_'+str(j)].append(np.percentile(s_clean, j))
    return g_percentile, s_percentile

def percentile(x, percent=[25,50,75,99]):
    dim3=np.size(x, axis=0)
    # initialize output variables
    x_percentile={}
    for j in percent:
        x_percentile[str(j)]=[]
        
    for i in range(0, dim3):
        x_clean=clean(x[i,:,:])
        for j in percent:
            x_percentile[str(j)].append(np.percentile(x_clean, j))
    return x_percentile

def mode(x):
    import statistics
    x=clean(x)
    x_mode=statistics.mode(x)
    return x_mode

def clean(g, s=np.nan):
    '''
    purge all zeros in g AND s and returns 1-D vectors
    g: vector or matrix
    '''
    if np.all(np.isnan(s))==True:
        #clean
        g_clean=g[g>0]
        #resize in 1-D vector
        g_clean=np.reshape(g_clean, (np.size(g_clean))) 
        return g_clean
    else:
        # clean
        g_clean=g[np.logical_and(g>0, s>0)]
        s_clean=s[np.logical_and(g>0, s>0)]
        # create 1-D vector
        g_clean=np.reshape(g_clean, (np.size(g_clean)))
        s_clean=np.reshape(s_clean, (np.size(s_clean)))
        return g_clean,s_clean

def quantize(histogram2d):
    histogram1d = np.reshape(histogram2d, np.size(histogram2d))
    hist, edges = np.histogram(histogram1d)
    hist_max=np.max(hist)
    N=50
    dN=hist_max/N
    for i in range (0, N):
        histogram2d=np.where(np.logical_and(histogram2d>i, histogram2d<=i+dN), (i+dN)/2, histogram2d)
    return histogram2d

def FindMainPeak(g, s):
    '''
    g_peak_coord, s_peak_coord, g_peak_freq, s_peak_freq=phasor.findpeaks(g,s)
    find local maxima coordinates and value
    '''
    dim3=np.size(g, axis=0)
    g_peak_coord=[]
    s_peak_coord=[]
    freq=[]
    for i in range(0, dim3):
        # clean data
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        # calculate histogram and extract peak position
        hist, g_edges, s_edges=histogram(g_clean, s_clean, bins=100)
        hist=hist[0,:,:]
        # normalize
        hist_max=np.max(hist)
        hist_norm=np.divide(hist, hist_max)
        # quantization
        # hist=quantize(hist)
        # blurring
        # hist_norm=filters.median(hist_norm)
        hist_norm=np.round(hist_norm, 2)
        # find peak
        hist_mask=hist_norm>0.9
        # hist_mask=h_maxima(hist_norm, 0.9)
        peak_positions=np.argwhere(hist_mask)
        allowed_freqs=hist[hist_mask]
        g_peak_position=np.sum(np.multiply(peak_positions[:,0], allowed_freqs))/np.sum(allowed_freqs)
        s_peak_position=np.sum(np.multiply(peak_positions[:,1], allowed_freqs))/np.sum(allowed_freqs)
        g_peak_position, s_peak_position=np.round([g_peak_position, s_peak_position], 0)
        g_peak_position, s_peak_position=np.int64([g_peak_position, s_peak_position])
        # g_peak_position=peak_positions[:,0]
        # s_peak_position=peak_positions[:,1]
        # convert peak position into coordinate
        g_edge_1=g_edges[0][g_peak_position]
        g_edge_2=g_edges[0][g_peak_position+1]
        s_edge_1=s_edges[0][s_peak_position]
        s_edge_2=s_edges[0][s_peak_position+1]
        coord_g=np.mean([g_edge_1, g_edge_2], axis=0)
        coord_s=np.mean([s_edge_1,s_edge_2], axis=0)
        if np.shape(coord_g)==0: 
            coord_g=np.nan
            coord_s=np.nan
            freq.append(np.nan)
        else:
            # set frequence related to peak of coordinate (g,s)
            freq.append(hist[g_peak_position, s_peak_position])
        # append current data to the final structure
        g_peak_coord.append(coord_g)
        s_peak_coord.append(coord_s)        
    # convert in numpy array
    # g_peak_coord=np.array(g_peak_coord)
    # s_peak_coord=np.array(s_peak_coord)
    # freq=np.array(freq)
    return g_peak_coord, s_peak_coord, freq

# def findpeaks_old(g, s):
#     '''
#     g_peak_coord, s_peak_coord, g_peak_freq, s_peak_freq=phasor.findpeaks(g,s)
#     find local maxima coordinates and value
#     '''
#     dim3=np.size(g, axis=0)
#     g_peak_coord=[]
#     s_peak_coord=[]
#     freq=[]
#     for i in range(0, dim3):
#         # clean data
#         g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
#         # calculate histogram and extract peak position
#         hist, g_edges, s_edges=histogram(g_clean, s_clean, bins=100)
#         hist=hist[0,:,:]
#         # normalize
#         hist_max=np.max(hist)
#         hist_normalized=np.divide(hist, hist_max)
#         # quantization
#         # hist=quantize(hist)
#         # blurring
#         hist=filters.median(hist)
#         # find peak
#         hist_mask=h_maxima(hist, 50)
#         peak_positions=np.argwhere(hist_mask)
#         g_peak_position=peak_positions[:,0]
#         s_peak_position=peak_positions[:,1]
#         # convert peak position into coordinate
#         g_edge_1=g_edges[0][g_peak_position]
#         g_edge_2=g_edges[0][g_peak_position+1]
#         s_edge_1=s_edges[0][s_peak_position]
#         s_edge_2=s_edges[0][s_peak_position+1]
#         coord_g=np.mean([g_edge_1, g_edge_2], axis=0)
#         coord_s=np.mean([s_edge_1,s_edge_2], axis=0)
#         if np.shape(coord_g)==0: 
#             coord_g=np.nan
#             coord_s=np.nan
#             freq.append(np.nan)
#         else:
#             # set frequence related to peak of coordinate (g,s)
#             freq.append(hist[0, g_peak_position, s_peak_position])
#         # append current data to the final structure
#         g_peak_coord.append(coord_g)
#         s_peak_coord.append(coord_s)        
#     # convert in numpy array
#     # g_peak_coord=np.array(g_peak_coord)
#     # s_peak_coord=np.array(s_peak_coord)
#     # freq=np.array(freq)
#     return g_peak_coord, s_peak_coord, freq

# def local_maxima(array, min_distance = 1, periodic=False, edges_allowed=True): 
#     """Find all local maxima of the array, separated by at least min_distance."""
#     array = np.asarray(array)
#     cval = 0 
#     if periodic: 
#         mode = 'wrap' 
#     elif edges_allowed: 
#         mode = 'nearest' 
#     else: 
#         mode = 'constant' 
#     cval = array.max()+1 
#     max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
#     return [indices[max_points] for indices in np.indices(array.shape)]

def histogram(g,s, bins=100):
    dim=np.shape(g)
    if np.size(dim)==1:
        dim3=1
        g=np.reshape(g, (1,1,dim[0]))
        s=np.reshape(s, (1,1,dim[0]))
    elif np.size(dim)==2:
        dim3=1
        g=np.reshape(g, (1, dim[0], dim[1]))
        s=np.reshape(s, (1, dim[0], dim[1]))
    else:
        dim3=np.size(g, axis=0)
    hist=[]
    xedges=[]
    yedges=[]
    for i in range(0, dim3):
        g_clean, s_clean=clean(g[i,:,:],s[i,:,:])
        hist_temp, xedges_temp, yedges_temp=np.histogram2d(g_clean, s_clean, bins=bins)
        hist.append(hist_temp)
        xedges.append(xedges_temp)
        yedges.append(yedges_temp)
    hist=np.array(hist)
    xedges=np.array(xedges)
    yedges=np.array(yedges)
    return hist, xedges, yedges
        
    
def create_dataframe(g, s, g_har2, s_har2):
    # # calculate phasor data
    # barycenter -> row mean
    g_baryc, s_baryc=barycenter(g,s)
    g_center, s_center=FindCenter(g, s)
    g_peak, s_peak, freq = FindMainPeak(g, s)
    computedcircularity=circularity(g, s)
    computedperimeter=perimeter(g, s)
    computedarea=area(g, s)
    
    g_har2_baryc, s_har2_baryc = barycenter(g_har2, s_har2)
    g_har2_center, s_har2_center=FindCenter(g_har2, s_har2)
    g_har2_peak, s_har2_peak, freq_har2 = FindMainPeak(g_har2, s_har2)
    computedcircularity_har2=circularity(g_har2, s_har2)
    computedperimeter_har2=perimeter(g_har2, s_har2)
    computedarea_har2=area(g_har2, s_har2)
    # find peaks
    # g_peak_coord, s_peak_coord, peak_freq=phasor.findpeaks(g,s)
    
    # center -> median of unduplicated data
    
    # Create a Dataframe for Machine Learning
    
    data={
         'g_barycenter': g_baryc,
         # 'g_peak_coordinate': g_peak_coord,
         'g_center': g_center,
         's_barycenter': s_baryc,
         # 's_peak_coordinate': s_peak_coord,
         's_center': s_center,
         'g_peak': g_peak,
         's_peak': s_peak,
         'circularity': computedcircularity,
         'perimeter': computedperimeter,
         'area': computedarea,
         
        ## second harmonic parameters
        'g_har2_barycenter': g_har2_baryc,
        # 'g_peak_coordinate': g_peak_coord,
        'g_har2_center': g_har2_center,
        's_har2_barycenter': s_har2_baryc,
        # 's_peak_coordinate': s_peak_coord,
        's_har2_center': s_har2_center,
        'g_har2_peak': g_har2_peak,
        's_har2_peak': s_har2_peak,
        'har2_circularity': computedcircularity_har2,
        'har2_perimeter': computedperimeter_har2,
        'har2_area': computedarea_har2
          
          }
    
    dataframe=pd.DataFrame(data)
    return dataframe

def OnlyNumeric_FromDataFrame(dataframe):
    dataframe = dataframe.select_dtypes(include=np.number)
    return dataframe
    
def ConfidenceInterval(x, CI=95):
    CI=np.array(CI)
    if np.any(CI<=0) or np.any(CI>=100):
        print('invalid CI')
    else:
        x=clean(x) #clean zeros and NANs
        x_CI={} # initialize output variable
        for j in CI:
            interval=np.divide(100-j, 2)
            x_CI['CI_'+str(j)+'_min']=np.percentile(x, interval)
            x_CI['CI_'+str(j)+'_max']=np.percentile(x, j+interval)
        return x_CI


#assess KClustering throug Elbow function and WCSS (within-cluster sums of squares) 
#def ElbowFunction(cluster):
#	wcss=[]
#	for i in range(1,5):
#	  kmeans = KMeans(i)
#	  kmeans.fit(cluster)
#	  wcss_iter = kmeans.inertia_
#	  wcss.append(wcss_iter)
#	  
#	number_clusters = range(1,5)
#	plt.plot(number_clusters,wcss)
#	plt.title('The Elbow title')
#	plt.xlabel('Number of clusters')
#	plt.ylabel('WCSS')
#  
##KMean Clustering
#def KMeanClustering(cluster):
#	kmeans=KMeans(2)
#	clustering=kmeans.fit(cluster)
#	plt.plot(phasor_plot.C_x,phasor_plot.C_y)
#	plt.xlim(0,1)
#	plt.ylim(0,0.7)
#	plt.scatter(phasor_plot.universal_x,phasor_plot.universal_y, c=clustering.labels_, cmap='rainbow')

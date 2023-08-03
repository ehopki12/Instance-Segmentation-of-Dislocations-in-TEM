from pathlib import Path
from ultralytics.DislocationMaskProcessing.preprocessing import MaskPreprocessor
from ultralytics.DislocationMaskProcessing.skeletonization import show_skeletonized_image, lee_skeletonization, separate_overlapping_segments
from ultralytics.DislocationMaskProcessing.curvefitting import fit_spline
from ultralytics.DislocationMaskProcessing.skeletonization import bounding_box
from ultralytics.DislocationMaskProcessing.curvefitting import order_data_points
import numpy as np 
import copy

def get_pileup(i, l):
    temp=[]
    count=0
    temp.append(i)
    while (count < 4):
        for j in l:
    #         print("before" , i, j, temp )
            if(i in j):
                l.remove(j)
                j.remove(i)
                if (j[0] not in temp): temp.append(j[0])
                i = j[0]
    #             print("after" , i, j, temp )

            count +=1
    return set(temp)

def get_endpoints(path_img , area_limits=(80,500),verbose=False,show_bboxes=False,):
        """
        Given a dislocation mask get 
        1. xnew, ynew: points of all dislocations present , xnew[0] , ynew[1] will give points of first dislocation
        2. lengths: length of all dislocations measured in pixels 
        3. Bounding box around each dislocations 
        """
        p = MaskPreprocessor(path_img,verbose=verbose,)
        # p.invert()
        p.thresholding()

        d_spline = {"Xs_2D_WCS" : [],
                    "Ys_2D_WCS" : []}
        loc_min = []
        loc_max = []
        labels, pixel_groups, bboxes, centers = p.connected_components(connectivity = 8,
                                                              area_limits = area_limits,
                                                              show_bboxes=show_bboxes, 
                                                              show_components=False, 
                                                              verbose=verbose)


        x_new , y_new,lengths = [] , [], []
        for i , pixel_group in enumerate(pixel_groups):
                img, x, y , wrong_start,length = lee_skeletonization(pixel_group,start=0) 
                if (wrong_start):
                    img, x, y , wrong_start,length = lee_skeletonization(pixel_group,start=-1)
                x_new.append(x)
                y_new.append(y)
                lengths.append(length)
                xlim, ylim = bounding_box(img)
                d_spline["Xs_2D_WCS"].append(x.tolist())
                d_spline["Ys_2D_WCS"].append((p.image.shape[0]-y).tolist())
                loc_max.append(x.max())
                loc_min.append(x.min())

        value , counts = np.unique(np.array(loc_max),return_counts=True)
        try:
            if (counts.max() > 1) : 
                loc = loc_min
            else:
                loc = loc_max
        except:
            loc = loc_max
            
        d_spline["Xs_2D_WCS"] = [x for _,x in sorted(zip(loc,d_spline["Xs_2D_WCS"]))] # sort than based on their locations 
        d_spline["Ys_2D_WCS"] = [x for _,x in sorted(zip(loc,d_spline["Ys_2D_WCS"]))]
        lengths = [x for _,x in sorted(zip(loc,lengths))]
        centers = [x for _,x in sorted(zip(loc,centers))]


        d_spline["Xs_2D_WCS"].reverse()
        d_spline["Ys_2D_WCS"].reverse()
        
        x_p = [[],[],[],[]]
        y_p = [[],[],[],[]]
            
        return x_new, y_new, lengths, x_p , y_p , centers




def get_endpoints2(path_img , area_limits=(100,1000),verbose=False,show_bboxes=False,):
        """
        Given a dislocation mask get 
        1. xnew, ynew: points of all dislocations present , xnew[0] , ynew[1] will give points of first dislocation
        2. lengths: length of all dislocations measured in pixels 
        3. Bounding box around each dislocations 
        """
        p = MaskPreprocessor(path_img,verbose=verbose,)
        # p.invert()
        p.thresholding()

        d_spline = {"Xs_2D_WCS" : [],
                    "Ys_2D_WCS" : []}
        loc_min = []
        loc_max = []
        labels, pixel_groups, bboxes, centers = p.connected_components(connectivity = 8,
                                                              area_limits = area_limits,
                                                              show_bboxes=show_bboxes, 
                                                              show_components=False, 
                                                              verbose=verbose)

        dis = []
        for i in range(len(centers)):
            c = centers.copy()
            c[i] = [100000,100000]
            xy = np.array(c)
            min_dis = np.linalg.norm(xy-[centers[i]],axis=1).min()
            if (min_dis < 100):
                dis.append([i,np.linalg.norm(xy-[centers[i]],axis=1).argmin()])

        l = copy.deepcopy(dis)
        pileups_list = []
        for i in range(len(centers)):
            t = list(get_pileup(i,copy.deepcopy(dis)))
            if (t not in pileups_list): pileups_list.append(t)

        x_new , y_new,lengths = [] , [], []
        for i , pixel_group in enumerate(pixel_groups):
                img, x, y , wrong_start,length = lee_skeletonization(pixel_group,start=0) 
                if (wrong_start):
                    img, x, y , wrong_start,length = lee_skeletonization(pixel_group,start=-1)
                x_new.append(x)
                y_new.append(y)
                lengths.append(length)
                xlim, ylim = bounding_box(img)
                d_spline["Xs_2D_WCS"].append(x.tolist())
                d_spline["Ys_2D_WCS"].append((p.image.shape[0]-y).tolist())
                loc_max.append(x.max())
                loc_min.append(x.min())

        value , counts = np.unique(np.array(loc_max),return_counts=True)
        if (counts.max() > 1) : 
            loc = loc_min
        else:
            loc = loc_max

        d_spline["Xs_2D_WCS"] = [x for _,x in sorted(zip(loc,d_spline["Xs_2D_WCS"]))] # sort than based on their locations 
        d_spline["Ys_2D_WCS"] = [x for _,x in sorted(zip(loc,d_spline["Ys_2D_WCS"]))]
        lengths = [x for _,x in sorted(zip(loc,lengths))]
        centers = [x for _,x in sorted(zip(loc,centers))]


        d_spline["Xs_2D_WCS"].reverse()
        d_spline["Ys_2D_WCS"].reverse()
        
        x_p = [[],[],[],[]]
        y_p = [[],[],[],[]]
        for i ,(x , y) in enumerate(zip(d_spline["Xs_2D_WCS"],d_spline["Ys_2D_WCS"])):
            y = [p.image.shape[0]-k for k in y]
            x = np.array(x)
            y = np.array(y)
            x_p[0].append(x[np.argmin(x)])
            y_p[0].append(y[np.argmin(x)])
            x_p[1].append(x[np.argmax(x)])
            y_p[1].append(y[np.argmax(x)])
            
            x_p[2].append(x[np.argmin(y)])
            y_p[2].append(y[np.argmin(y)])
            x_p[3].append(x[np.argmax(y)])
            y_p[3].append(y[np.argmax(y)])
            
            
        return x_new, y_new, lengths, x_p , y_p , centers
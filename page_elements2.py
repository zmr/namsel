#! /usr/bin/python
# encoding: utf-8
'''Page Elements'''

import cv2 as cv
import numpy as np
from sklearn.mixture import GMM
from scipy.stats import mode as statsmode
from classify import label_chars, load_cls
from scipy.ndimage.interpolation import rotate
from utils import invert_bw
from feature_extraction import normalize_and_extract_features
from fast_utils import to255
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmin
from scipy.interpolate import UnivariateSpline, splrep, splev
from collections import OrderedDict

class PageElements(object):
    '''Page Elements object - a representation of the tiff image as a set
    of elements (contours, bounding boxes) and measurements used for recognition
    
    Parameters:
    -----------
    img_arr: 2d numpy array containing pixel data of the image
    
    small_coef: int, default=2
        A scalar value used in filtering out small ("noise") objects in the
        image.
        
        This may be deprecated soon. It is useful in situations where you
        know the typeset being used and want to ensure filtering is not too
        lax or aggressive.
        
    Attributes:
    ------
    contours: list, a list of contours return by cv.findContours
    
    hierarchy: list, contour hierarchy exported by cv.findContours
    
    boxes: list, list of bounding boxes for the page
    
    indices: list, list of integers representing the indices for contours and
        boxes that have not been filtered
    
    char_mean, char_std, tsek_mean, tsek_std: float, parameters of the Gaussian
        distributions for letters and punctuation on the page (first pass)
    
    page_array: 2d array of containing newly drawn image with filtered blobs
        removed
    
    Methods:
    --------
    char_gaussians: class method for using 2 class GMM
    
    get_tops: helper function for getting the top y coordinates of all
        bounding boxes on the page (-filter boxes)
    '''
    
    def __init__(self, img_arr, fast_cls, small_coef=1, low_ink=False, \
                 page_type=None, flpath=None, detect_o=True,\
                 clear_hr = False): #lower coef means more filtering USE 3 for nying gyud
        self.img_arr = img_arr
        self.page_type = page_type
        self.flpath = flpath
        self.low_ink = low_ink
        self.detect_o = detect_o

        self.cached_features = OrderedDict()
        self.cached_pred_prob = OrderedDict()
        self._contour_mode = cv.RETR_TREE
        ### repeatedly called functions
        ones = np.ones
        uint8 = np.uint8
        predict = fast_cls.predict
        predict_proba = fast_cls.predict_proba
        self.contours, self.hierarchy = self._contours()
        self.boxes = []
        self.indices = []
        self.small_coef = small_coef
        
        FILTERED_PUNC = (u'།', u'་', u']', u'[')
        
        self._set_shape_measurements()
        if page_type == 'pecha':
            if clear_hr:
                print 'Warning: clear_hr called on pecha format. For clearing text'
                self.force_clear_hr()
            self.set_pecha_layout()
            if self.indices:
                content_parent = int(statsmode([self.hierarchy[0][i][3] for i in self.indices])[0])
            else:
                print 'no content found'
        else:

            content_parent = int(statsmode([hier[3] for hier in self.hierarchy[0]])[0])
            self.indices = self.get_indices()

        outer_contours = []
        outer_widths = []

        
        ## Iterate through all contours
        for i in self.indices:
            cbox = self.get_boxes()[i]
            x,y,w,h = cbox
             ### THIS SECOND CONDITION IS CAUSING A LOT OF PROBLEMS. Recently 
             # added the len(indices) < 40 as a way to prevent exaggerated
             # filtering of small lines where gaussian width measures
             # are meaningless due to small sample size (too few contours)
            if self.hierarchy[0][i][3] == content_parent and (cbox[2] < .1*self.img_arr.shape[1] or len(self.indices) < 40 ): 
#            if self.hierarchy[0][i][3] == content_parent and cbox[2] < 3*self.char_mean:  ### THIS SECOND CONDITION IS CAUSING A LOT OF PROBLEMS
                outer_contours.append(i)
                outer_widths.append(cbox[2])
            else:
                if cbox[2] > .66*self.img_arr.shape[1]:
                    print cbox[2] / float(self.img_arr.shape[1])
                if clear_hr and .995*self.img_arr.shape[1] > cbox[2] > \
                .66*self.img_arr.shape[1] and cbox[1] < .25*self.img_arr.shape[0]:
                    self.img_arr[0:cbox[1]+cbox[3], :] = 1
#                 print 'rejected box. too wide?', cbox[2] >= .1*self.img_arr.shape[1] 
        width_measures = self.char_gaussians(outer_widths)
        
        for i,j in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, i, j)

        self.small_contour_indices = []
        self.indices = [] # Need to reset!19
        self.emph_symbols = []
        self.naros = []
        for i in outer_contours:
            cbox = self.get_boxes()[i]
            # if small and has no children, put in small list (this could backfire with false interiors e.g. from salt and pepper noise)
            ## NOTE: previously small was defined as less than tsek_mean + 3xtsek std
            ## however, this wasn't always working. changing to less than charmean
            ## minus 2xchar std however should watch to see if is ok for many different inputs...
            
            x,y,w,h = cbox
            tmparr = ones((h,w), dtype=uint8)
            tmparr = self.draw_contour_and_children(i, tmparr, (-x,-y))

            features = normalize_and_extract_features(tmparr)
            self.cached_features[i] = features
            
            prprob = predict_proba(features)
            
            mxinx = prprob.argmax()
            quick_prd = label_chars[mxinx]
            self.cached_pred_prob[i] = (mxinx, prprob[0])
            
            is_emph_symbol = quick_prd in set([u'༷', u'༵', u'༼', u'༽'])
            if is_emph_symbol:
                self.emph_symbols.append(i)
                
                print 'EMPHSYMBOLFOUND', quick_prd
            elif quick_prd == u'ོ' and self.detect_o:
                self.naros.append(i)
                
            elif cbox[2] < 7:
                
                continue
            elif (cbox[2] <= self.tsek_mean*3 and 
                quick_prd in FILTERED_PUNC) and not self.low_ink: # default!!!
                self.small_contour_indices.append(i)
            else:
                self.indices.append(i)
        if self.detect_o:
            print 'pre-filtered na-ro vowel', len(self.naros), 'found'    
        

        if self.low_ink:
            self._low_ink_setting()
    
    
    def force_clear_hr(self):
        boxes = self.get_boxes()
        for cbox in boxes:
            if .995*self.img_arr.shape[1] > cbox[2] > \
                    .66*self.img_arr.shape[1] and cbox[1] < .25*self.img_arr.shape[0]:
                        self.img_arr[0:cbox[1]+cbox[3], :] = 1

    def _low_ink_setting(self):
        print 'IMPORTANT: Low ink setting=True'
        a = self.img_arr.copy()*255
        
        ############## Effects
        #**Default**#
#         erode_iter = 3
#         vertblur = 15
#         horizblur = 1
#         threshold = 170
          
        #**mild vertical blurring**#
#         erode_iter = 1
#         vertblur = 5
#         horizblur = 1
#         threshold = 127
        
        #**mild vertical blurring**#
        #**mild vertical blurring**#
        
        #**other**#
        erode_iter = 2
        vertblur = 35
        horizblur = 1
        threshold = 160
        
        
        #############
        
        
        a = cv.erode(a, None, iterations=erode_iter)
        ##### parameters below are highly text-dependent unfortunately...
#         a = cv.blur(a, (9,61))
#         a = cv.blur(a, (9,61))
#         a = cv.blur(a, (int(.5*self.tsek_mean),int(3*self.tsek_mean)))
#         a = cv.blur(a, (1,15))
        a = cv.blur(a, (horizblur,vertblur))
#         a = cv.blur(a, (15,1))
#         a = cv.blur(a, (9,70))
#         a = cv.blur(a, (1,50))
#         ret, a = cv.threshold(a, 175, 255, cv.THRESH_BINARY)
        ret, a = cv.threshold(a, threshold, 255, cv.THRESH_BINARY)
#         ret, a = cv.threshold(a, 200, 255, cv.THRESH_BINARY)
#         ret, a = cv.threshold(a, 160, 255, cv.THRESH_BINARY)
        ctrs, hier = cv.findContours(a, mode=self._contour_mode , 
                                     method=cv.CHAIN_APPROX_SIMPLE)
        
        self.low_ink_boxes = [cv.boundingRect(c) for c in ctrs]
        self.low_ink_boxes = [i for i in self.low_ink_boxes if 
                              i[2] < 1.33*self.char_mean]
        del a, ctrs, hier
        
    def _contours(self):

        return cv.findContours(self.img_arr.copy(), mode=self._contour_mode, 
                               method=cv.CHAIN_APPROX_SIMPLE)

    
    def get_boxes(self):
        '''Retrieve bounding boxes. Create them if not yet cached'''
        if not self.boxes:
            self.boxes = self._boxes()
       
        return self.boxes
    
    def _boxes(self):
        return [cv.boundingRect(c) for c in self.contours]
    
    def get_indices(self):
        if not self.indices:
            self.indices = [i for i, b in enumerate(self.get_boxes())] #if (
        return self.indices
        
    def _set_shape_measurements(self):
        width_measures = self.char_gaussians([b[2] for b in self.get_boxes() if
                                               b[2] < .1*self.img_arr.shape[1]])
        for i,j in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, i, j)
    

    def update_shapes(self):
        self.contours, self.hierarchy = self._contours()
        self.boxes = self._boxes()
        self._set_shape_measurements()
        self.indices = [i for i, b in enumerate(self.get_boxes()) if (
               max(b[2], b[3]) <= 6 * self.char_mean )] 

    
    def _draw_new_page(self):
        self.page_array = np.ones_like(self.img_arr)
        
        self.tall = set([i for i in self.get_indices() if 
                         self.get_boxes()[i][3] > 3*self.char_mean])
        
        cv.drawContours(self.page_array, [self.contours[i] for i in 
                        range(len(self.contours)) if 
                        self.get_boxes()[i][2] > self.smlmean + 3*self.smstd], 
                        -1,0, thickness = -1)
        import Image
        Image.fromarray(self.page_array*255).show()
        
    @classmethod
    def char_gaussians(cls, widths):
        
        widths = np.array(widths)
        widths.shape = (len(widths),1)
        cls.median_width = np.median(widths)
        
        gmm = GMM(n_components = 2, n_iter=100)
        try:
            gmm.fit(widths)
        except ValueError:
            return (0,0,0,0)
        means = gmm.means_
        stds = np.sqrt(gmm.covars_)
        cls.gmm = gmm
        char_mean_ind = np.argmax(means)
        char_mean = float(means[char_mean_ind]) # Page character width mean
        char_std = float(stds[char_mean_ind][0]) # Page character std dev
        
        cls.tsek_mean_ind = np.argmin(means)
        
        tsek_mean = float(means[cls.tsek_mean_ind])
        tsek_std = float(stds[cls.tsek_mean_ind][0])
        return (char_mean, char_std, tsek_mean, tsek_std)


    def get_tops(self):       
        return [self.get_boxes()[i][1] for i in self.get_indices()]
    
    def draw_contour_and_children(self, root_ind, char_arr=None, offset=()):
        char_contours = [root_ind]
        root = self.hierarchy[0][root_ind]
        if root[2] >= 0:
            char_contours.append(root[2]) # add root's first child
            child_hier = self.hierarchy[0][root[2]] # get hier for 1st child
            if child_hier[0] >= 0: # if child has sib, continue to loop
                has_sibling = True
            else: has_sibling = False # ... else skip loop and draw
            
            while has_sibling:
                ind = child_hier[0] # get sibling's index
                char_contours.append(ind) # add sibling's index
                child_hier = self.hierarchy[0][ind] # get sibling's hierarchy
                if child_hier[0] < 0: # if sibling has sibling, continue loop
                    has_sibling = False
        
        if not hasattr(char_arr, 'dtype'):
            char_box = self.get_boxes()[root_ind]
            x,y,w,h = char_box
            char_arr = np.ones((h,w), dtype=np.uint8)
            offset = (-x, -y)
        cv.drawContours(char_arr, [self.contours[j] for j in char_contours], -1,0, thickness = -1, offset=offset)
        return char_arr
    
    def detect_num_lines(self, content_box_dict):
        '''content_box_dict has values {'chars':[], 'b':b, 'boxes':[], 
                                'num_boxes':0, 'num_chars':0}
        
        where chars are the indices of chars in the content box, b is the 
        the xywh dimensions of the box, boxes are the sub-boxes of the 
        document tree contained in this box (not box chars but large page-
        structuring boxes. 
        
        Note: page_type must be set to "pecha"
        '''
        
        cbx, cby, cbw, cbh = content_box_dict['b']
              
        cbox_arr = np.ones((cbh, cbw), dtype=self.img_arr.dtype)
        
        tsekmeanfloor = np.floor(self.tsek_mean)
        tsekstdfloor = np.floor(self.tsek_std)
        cv.drawContours(cbox_arr, [self.contours[i] for i in content_box_dict['chars']
                        if ((self.get_boxes()[i][2] > 
                        (tsekmeanfloor - 
               self.small_coef * tsekstdfloor)  or 
               self.get_boxes()[i][2] < .1*self.img_arr.shape[1]) and 
                            self.get_boxes()[i][3] > 10) 
                                   ], -1, 0, thickness=-1, offset=(-cbx, -cby))
        cbox_arr = cbox_arr[5:-5, :] # shorten from the top and bottom to help out trim in the event of small noise
        cbox_arr = cv.erode(cbox_arr, None, iterations=5)
        cbox_arr = to255(cbox_arr)

        cv.blur(cbox_arr, (75, 19), dst=cbox_arr)

        ####################
#         print 'warning: using non default (127) line count threshold'
#         ret, cbox_arr = cv.threshold(cbox_arr, 127, 1, cv.THRESH_BINARY)
        ####################
        ret, cbox_arr = cv.threshold(cbox_arr, 200, 1, cv.THRESH_BINARY) #DEFAULT!
        ###################

        vsum = cbox_arr.sum(axis=1)

        vsum_smoothed = gaussian_filter1d(vsum, 25) ###DEFAULT
#         vsum_smoothed = gaussian_filter1d(vsum, 13)
        len_vsum = len(vsum)

        fx = UnivariateSpline(range(len_vsum), vsum_smoothed)
        tck = splrep(range(len_vsum), fx(range(len_vsum)))
        y = splev(range(len_vsum), tck, der=1)
        tck = splrep(range(len_vsum), y)

        mins = argrelmin(fx(range(len_vsum)))
#       
        ### Filter false peaks that show up from speckles on page
        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .01]
#        
        self.num_lines = len(mins)

    
    def draw_hough_outline(self, arr):
        
        arr = invert_bw(arr)
        h = cv.HoughLinesP(arr, 2, np.pi/4, 1, minLineLength=arr.shape[0]*.15, maxLineGap=5) #This
        PI_O4 = np.pi/4
        if h is not None:
            for line in h[0]:
                new = (line[2]-line[0], line[3] - line[1])
                val = (new[0]/np.sqrt(np.dot(new, new)))
                theta = np.arccos(val)
                if theta >= PI_O4: # Vertical line
                    if line[0] < .5*arr.shape[1]:
                        arr[:,:line[0]+12] = 0
                    else:
                        arr[:,line[0]-12:] = 0
                else: # horizontal line
                    if line[2] - line[0] >= .15 * arr.shape[1]:
#                         cv.line(arr, (0, line[1]), (arr.shape[1], line[1]), 1, thickness=50)
                        if line[1] < .5 *arr.shape[0]:
                            arr[:line[1]+17, :] = 0
                        else:
                            arr[line[1]-5:,:] = 0
        

        return ((arr*-1)+1).astype(np.uint8)

    def save_margin_content(self, tree, content_box):
        '''Look at margin content and try to OCR it. Save results in a pickle
        file of a dictionary object:
        d = {'left':['margin info 1', ...], 'right':['right margin info 1', etc]}
        
        Margin content is tricky since letters are often not defined as well
        as the main page content. The current OCR implementation also stumbles
        on text with very few characters. Page numbers don't do well for some
        reason...
        '''
        
        import cPickle as pickle
        import os
        content_box_right_edge = tree[content_box]['b'][0] + tree[content_box]['b'][2]
        inset = 20

        right_content = []
        left_content = []
        for brnch in tree:
            if brnch != content_box:
                outer_box = brnch

                if tree[outer_box]['num_chars'] != 0:
                    bx = tree[outer_box]['b']
                    arr = self.img_arr[bx[1]+inset:bx[1]+bx[3]-inset, bx[0]+inset:bx[0]+bx[2]-inset]

                    text = ''
                    if bx[0] > content_box_right_edge:
                        arr = rotate(arr, -90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            right_content.append(text)
                    else:
                        arr = rotate(arr, 90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            left_content.append(text)
        pklname = os.path.join(os.path.dirname(self.flpath), os.path.basename(self.flpath)[:-4]+'_margin_content.pkl')
        pickle.dump({'right':right_content, 'left':left_content}, open(pklname, 'wb'))
#        import sys; sys.exit()
    
    def set_pecha_layout(self):
        a = self.img_arr.copy()
        
        if self.img_arr.shape[1] > 2*self.img_arr.shape[0]:
            self._page_type = 'pecha'
        else:
            self._page_type = 'book'
        
        if self._page_type == 'pecha': # Page is pecha format
            a = self.draw_hough_outline(a)
            
        self.img_arr = a.copy()
        self.update_shapes()
        
        a = cv.GaussianBlur(a, (5, 5), 0)
        contours, hierarchy = cv.findContours(a.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        
        
        ## Most of this logic for identifying rectangles comes from the 
        ## squares.py sample in opencv source code.
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
            return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
        
        border_boxes = []
        
        for j,cnt in enumerate(contours):
            cnt_len = cv.arcLength(cnt, True)
            orig_cnt = cnt.copy()
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], 
                                            cnt[(i+1) % 4], cnt[(i+2) % 4] ) 
                                  for i in range(4)])
                if max_cos < 0.1:
#                    print 'got one at %d' % j
#                    n = np.ones_like(a)
                    b = cv.boundingRect(orig_cnt)
#                     if self.clear_hr:
#                         print 'Warning: you are clearing text on a pecha page'
#                         self.img_arr[0:b[1]+b[3], :] = 1
                    x,y,w,h = b
                    border_boxes.append(b)
        border_boxes.sort(key=lambda b: (b[0],b[1]))
        
        def get_edges(b):
            l = b[0]
            r = b[0] + b[2]
            t = b[1]
            b = b[1] + b[3]
            return (l,r,t,b)

        def bid(b):
            return '%d-%d-%d-%d' % (b[0],b[1],b[2],b[3])
       
        tree = {}
        for b in border_boxes:
            tree[bid(b)] = {'chars':[], 'b':b, 'boxes':[], 'num_boxes':0, 'num_chars':0}    
        
        def b_contains_nb(b,nb):
            l1,r1,t1,b1 = get_edges(b)
            l2,r2,t2,b2 = get_edges(nb)
            return l1 <= l2 and r2 <= r1 and t1 <= t2 and b1 >= b2
            
        for i, b in enumerate(border_boxes):
            bx,by,bw,bh = b
            self.img_arr[by:by+1,bx+3:bx+bw-3] = 1
            self.img_arr[by+bh,by+bh-1:bx+3:bx+bw-3] = 1
            for nb in border_boxes[i+1:]:
                if b_contains_nb(b, nb):
                    tree[bid(b)]['boxes'].append(bid(nb))
                    tree[bid(b)]['num_boxes'] = len(tree[bid(b)]['boxes'])
        
        self.update_shapes()
        
        tree_keys = tree.keys()
        tree_keys.sort(key=lambda x: tree[x]['num_boxes'])
                
        ## Assign contours to boxes
        for i in self.get_indices():
            for k in tree_keys:
                box = tree[k]
                b = box['b']
                
                char_box = self.get_boxes()[i]
                if b_contains_nb(b, char_box):
                    tree[k]['chars'].append(i)
                    tree[k]['num_chars'] = len(tree[k]['chars'])
                    break
        
        def qualified_box(bx):
            '''Helper function that ignores boxes that contain other boxes.
            This is useful for finding the main content box which should
            be among the innermost boxes that have no box children '''
            
            if tree[bx]['num_boxes'] == 0:
                return tree[bx]['num_chars']
            else:
                return -1
        content_box = max(tree, key=qualified_box)
        self.indices = [i for i in tree[content_box]['chars'] if self.boxes[i][2] >= 7] 
        
        
        self.detect_num_lines(tree[content_box])

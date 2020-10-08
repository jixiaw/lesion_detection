# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:34:30 2019

@author: wjcongyu
"""

from models.candidet_2dcenternet import Candidet_2dcenternet
from models.resnet3d_sa import ResnetSelfAtten as NoduleFpr_sa
from models.resnet3d_wa import ResnetWithoutAtten as NoduleFpr_wa
from data_process.data_processor import resize
from data_process.data_processor import hu2gray
from data_process.data_processor import extractCube
from data_process.data_processor import get_convex_bbox_frm_3dmask
from data_process.nms.nms_wrapper import nms2d_3dinput as nms
from data_process.nms.nms_wrapper import nms_3d
import scipy.ndimage as ndimage
import cv2
import numpy as np
import os.path as osp
import imageio

class NoduleDetector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_error = ''
        self.is_ready = False
        self.candi_model = None
        self.fpr_model = None
        self.fpr2_model = None
        
    def initialize(self):
        '''
        Create session and Loading models and weights from disk. You have to call this func before
        calling classify
        '''
        candi_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, 'nodule_candi_det')
        assert osp.exists(candi_dir), 'no model dir found:' + candi_dir
        fpr_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, 'nodule_fpr')    
        assert osp.exists(fpr_dir), 'no model dir found:' + fpr_dir
        fpr2_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, 'nodule_fpr2')    
        assert osp.exists(fpr2_dir), 'no model dir found:' + fpr2_dir
       
        cls_dir = osp.join(self.cfg.CHECKPOINTS_ROOT, 'nodule_typecls')    
        assert osp.exists(cls_dir), 'no model dir found:' + cls_dir
        try:        
            self.candi_model = Candidet_2dcenternet(self.cfg.CANDI_INPUT_SHAPE, is_training=False, config= self.cfg, num_classes = 1, model_dir = candi_dir)
            checkpoint = self.candi_model.find_last()
            print (checkpoint)
            assert osp.exists(checkpoint), 'no checkpoint found:' + checkpoint
            self.candi_model.load_weights(checkpoint)
            
            self.fpr_model = NoduleFpr_sa([32,32,32], is_training=False, config= self.cfg, n_groups=4,n_blocks_per_group=[2,2,2,2], num_classes = 2, model_dir = fpr_dir)
            checkpoint = self.fpr_model.find_weights_of_last()
            print (checkpoint)
            assert osp.exists(checkpoint), 'no checkpoint found:' + checkpoint
            self.fpr_model.load_weights(checkpoint)
            
            self.fpr2_model = NoduleFpr_sa(self.cfg.FPR2_INPUT_SHAPE,is_training=False, config= self.cfg, n_groups=4,n_blocks_per_group=[2,2,3,3], num_classes = 2, model_dir = fpr2_dir)
            checkpoint = self.fpr2_model.find_weights_of_last()
            print (checkpoint)
            assert osp.exists(checkpoint), 'no checkpoint found:' + checkpoint
            self.fpr2_model.load_weights(checkpoint)
            
                      
            self.is_ready = True
            return True                 
                  
        
        except (OSError, TypeError) as reason:
            self._record_error(str(reason))
            self.is_ready = False
            return False
            
    def detect(self, volume, spacing, lung_mask_vol=None, debug=False):
        if debug:
            print('candidate dectection...')
        nodules = None
        candidates = self.detect_candidates(volume, spacing, lung_mask_vol)
            
        if (candidates is not None) and (candidates.shape[0]>0):
            if debug:
                print('candidates:', candidates.shape[0])
                print('false positive redutction stage 1...')
                
            nodules = self.reduce_false_positives(candidates, volume, spacing)
            
        if (nodules is not None) and (nodules.shape[0]>0):
            if debug:     
                print('candidates:', nodules.shape[0])
                print('false positive redutction stage 2...')                
           
            nodules = self.reduce_false_positives2(nodules, volume, spacing, self.cfg.FPR2_INPUT_SHAPE,self.cfg.test_fpr2_thres)
            
           
        if (nodules is not None) and nodules.shape[0]>1:
            keep = nms_3d(nodules, 0.01)
            nodules = nodules[keep]
        return nodules
    
    def detect_candidates(self, volume, spacing, lung_mask_vol=None):
        self.last_error = ''
        if not self.is_ready:
            self._record_error('detect_candidates:model not ready for nodule candidate detection!')
            return None
        
        if volume is None:
            self._record_error('detect_candidates:none volume data not allowed!')
            return None
        
        if len(volume.shape)!=3:
            self._record_error('detect_candidates:volume shape must equals 3! depthxheights x widths')
            return None
        oD,oH,oW = volume.shape
        
        #if lung_mask_vol is given, detection on lung regions
        lung_volume_bbox=[0,0,0,oW,oH,oD]
        if lung_mask_vol is not None and lung_mask_vol.shape == volume.shape:
            lung_volume_bbox = get_convex_bbox_frm_3dmask(lung_mask_vol)
            
        if lung_volume_bbox is None:
            lung_volume_bbox=[0,0,0,oW,oH,oD]
            
        lx1, ly1,lz1,lx2,ly2,lz2 = lung_volume_bbox
        if lx2-lx1<40 or ly2-ly1<40 or lz2-lz1<20:
            lung_volume_bbox = [0,0,0,oW,oH,oD]
            lx1, ly1,lz1,lx2,ly2,lz2 = lung_volume_bbox
            
        
        lung_volume = volume.copy()[lz1:lz2,ly1:ly2,lx1:lx2]
        feed_volume = self.convert2feedimageofcandi(lung_volume, self.cfg)
        
        #predict on batch
        nfeeds = int(feed_volume.shape[0]/self.cfg.test_feed_batchsize) + 1
        center_preds = []
        size_preds = []
        for k in range(nfeeds):
            l = k*self.cfg.test_feed_batchsize
            r = min(feed_volume.shape[0] , (k+1)*self.cfg.test_feed_batchsize)
            feed_batch = feed_volume[l:r]
            cnt_preds, sze_preds = self.candi_model.predict_on_batch(feed_batch)
            center_preds.append(cnt_preds)
            size_preds.append(sze_preds)

        cnt_preds = np.concatenate(center_preds, axis=0)
        size_preds = np.concatenate(size_preds, axis=0)

        #get predicted bboxes
        total_pred_bboxes = []
        D, H, W = lung_volume.shape
        nD, nH, nW, nC = cnt_preds.shape
        scale_d = D * 1.0 / nD
        scale_h = H * 1.0 / nH
        scale_w = W * 1.0 / nW
        if lung_mask_vol is not None and lung_mask_vol.shape == volume.shape:
            lung_mask = lung_mask_vol.copy()[lz1:lz2,ly1:ly2,lx1:lx2]
            
        
        for k in range(cnt_preds.shape[0]):
            cnt_pred = cnt_preds[k, ...]
            sze_pred = size_preds[k, ...]
            
            peaks = ndimage.maximum_filter(cnt_pred, size=(3, 3, 1))
            thres = 0.02
            labels, num_labels = ndimage.label(peaks > thres)
            coords = np.array(
                ndimage.measurements.center_of_mass(cnt_pred, labels=labels, index=np.arange(1, num_labels + 1)))
            scores = np.array(ndimage.measurements.maximum(cnt_pred, labels=labels, index=np.arange(1, num_labels + 1)))

            top_k = 20
            top_k_idx = scores.argsort()[::-1][0:top_k]

            top_k_coords = coords[top_k_idx, ...]
            top_k_scores = scores[top_k_idx]
            top_k_h = np.int32(np.abs(np.array([sze_pred[y, x, 0] for y, x, c in np.int32(top_k_coords)])))
            top_k_w = np.int32(np.abs(np.array([sze_pred[y, x, 1] for y, x, c in np.int32(top_k_coords)])))

            k_pred_bboxes = []
            for i in range(top_k_coords.shape[0]):
                y, x = top_k_coords[i][0:2]
                h = top_k_h[i]
                w = top_k_w[i]

                y = y * scale_h
                x = x * scale_w
                z = k * scale_d
                h = min(64, max(3, h * scale_h * 0.5))
                w = min(64, max(3, w * scale_w * 0.5))
                d = min(16, max(2, 0.5*max(w,h)*(spacing[0]/spacing[2])))
               
                x1 = int(max(0, x - w)) 
                x2 = int(min(W, x + w)) 
                y1 = int(max(0, y - h)) 
                y2 = int(min(H, y + h)) 
                z1 = int(max(0, z - d)) 
                z2 = int(min(D, z + d)) 
                              
                if lung_mask_vol is not None and lung_mask_vol.shape == volume.shape:
                    patch = lung_mask[z1:z2,y1:y2,x1:x2]
                    if np.count_nonzero(patch)<0.1*(patch.shape[0]*patch.shape[1]*patch.shape[2]):
                        continue                    
                k_pred_bboxes.append([x1, y1, z1, x2, y2, z2, top_k_scores[i]])
                
            if len(k_pred_bboxes) == 0:
                continue
            k_pred_bboxes = np.array(k_pred_bboxes, dtype=np.float32)
            keep = nms(k_pred_bboxes, 0.000001)
            k_pred_bboxes = k_pred_bboxes[keep, ...]
            total_pred_bboxes.append(k_pred_bboxes)

        if len(total_pred_bboxes)==0:
            return None
        total_pred_bboxes = np.concatenate(total_pred_bboxes, axis=0)
        
        total_pred_bboxes[:,0]+=lx1
        total_pred_bboxes[:,1]+=ly1
        total_pred_bboxes[:,2]+=lz1
        total_pred_bboxes[:,3]+=lx1
        total_pred_bboxes[:,4]+=ly1
        total_pred_bboxes[:,5]+=lz1
       
        return total_pred_bboxes
    
    def reduce_false_positives(self, candidates, volume, spacing):
        self.last_error = ''
        if not self.is_ready:
            self._record_error('reduce_false_positives:model not ready for nodule false positive reduction!')
            return None
        
        if volume is None:
            self._record_error('reduce_false_positives:none volume data not allowed!')
            return None
        
        if len(volume.shape)!=3:
            self._record_error('reduce_false_positives:volume shape must equals 3! depthxheights x widths')
            return None
        
        if candidates is None:
            self._record_error('reduce_false_positives:none candidates!')
            return None
        if candidates.shape[0] == 0:
            return candidates
     
       
        patches = []
        patch_pos_record = []
        for k in range(candidates.shape[0]):
            x1,y1,z1,x2,y2,z2,score = candidates[k,...]
            x = int(0.5*(x1+x2))
            y = int(0.5*(y1+y2))
            z = int(0.5*(z1+z2))
            w = x2-x1
            h = y2-y1
            d = max(w, h)
            d = d*3 if d<12 else d*2
                        
            d =  d*spacing[0]
            if d<self.cfg.min_crop_size:
                d=self.cfg.min_crop_size
            
            patch = extractCube(volume, [x,y,z], d, spacing, -1)
            patch = hu2gray(patch, WL=-600, WW=1200)
            patch = resize(patch, [32,32,32])
            minv = np.min(patch)
            maxv = np.max(patch)
            patch = (patch-minv)/(0.00001+maxv-minv)
            patch = patch.reshape(patch.shape + (1,))    
            patches.append(patch)
            patch_pos_record.append([x1,y1,z1,x2,y2,z2])
            
        patches = np.array(patches) 
        patch_pos_record =np.array(patch_pos_record)
        N = patches.shape[0]//self.cfg.test_fpr_feed_batchsize
        y_preds = []
       
        for i in range(N+1): 
            l = i*self.cfg.test_fpr_feed_batchsize
            r = min(patches.shape[0],(i+1)*self.cfg.test_fpr_feed_batchsize)
            patches_feed = patches[l:r,...]
            if patches_feed.shape[0]==0:
                continue
            y_pred = self.fpr_model.predict_on_batch(patches_feed)
            y_preds.append(y_pred)
            
        y_preds = np.concatenate(y_preds, axis=0)
        y_preds = y_preds[:,1]
        
        keep = np.where(y_preds>0.02)[0]
        scores = y_preds[keep]
        positions = patch_pos_record[keep,...]
       
        scores = np.expand_dims(scores, axis=-1)
       
        rst_nodules = np.concatenate([positions, scores],axis=-1)
        
       
        return rst_nodules    
            
    def reduce_false_positives2(self, candidates, volume, spacing, feed_size, thres):
        self.last_error = ''
        if not self.is_ready:
            self._record_error('reduce_false_positives2:model not ready for nodule false positive reduction!')
            return None
        
        if volume is None:
            self._record_error('reduce_false_positives2:none volume data not allowed!')
            return None
        
        if len(volume.shape)!=3:
            self._record_error('reduce_false_positives2:volume shape must equals 3! depthxheights x widths')
            return None
        
        if candidates is None:
            self._record_error('reduce_false_positives2:none candidates!')
            return None
        
        if candidates.shape[0] == 0:
            return candidates     
       
        patches = []
        patch_pos_record = []
        
            
        for k in range(candidates.shape[0]):
            x1,y1,z1,x2,y2,z2,score = candidates[k,...]
            x = int(0.5*(x1+x2))
            y = int(0.5*(y1+y2))
            z = int(0.5*(z1+z2))
           
            w = x2-x1
            h = y2-y1
            d = 0.5*(w+h) 
            d = 6 if d<6 else d
            if d> 50:
                continue
            crop_d_mm = d * 2.4 *spacing[0]
           
            
            patch = extractCube(volume, [x,y,z], crop_d_mm, spacing, -1)
            #patch = hu2gray(patch, WL=-500, WW=1000)
            patch = resize(patch, feed_size)            
            '''minv = np.min(patch)
            maxv = np.max(patch)
            patch = (patch-minv)/(0.00001+maxv-minv)'''
            
            patch = patch.reshape(patch.shape + (1,))    
            patches.append(patch)
            patch_pos_record.append([x1,y1,z1,x2,y2,z2])
            
        patches = np.array(patches) 
        patch_pos_record =np.array(patch_pos_record)
        N = patches.shape[0]//self.cfg.test_fpr_feed_batchsize
        y_preds = []
    
        for i in range(N+1):
            l = i*self.cfg.test_fpr_feed_batchsize
            r = min(patches.shape[0],(i+1)*self.cfg.test_fpr_feed_batchsize)
            patches_feed = patches[l:r,...]
           
            if patches_feed.shape[0]==0:
                continue
           
            y_pred = self.fpr2_model.predict_on_batch(patches_feed)
            y_preds.append(y_pred)
            
        y_preds = np.concatenate(y_preds, axis=0)
        y_preds = y_preds[:,1]
        
                    
        keep = np.where(y_preds>=thres)[0]
        scores = y_preds[keep]
        positions = patch_pos_record[keep,...]
       
        scores = np.expand_dims(scores, axis=-1)
      
        rst_nodules = np.concatenate([positions, scores],axis=-1)
        
        if rst_nodules.shape[0]>0:        
            mean_HUs = np.zeros((rst_nodules.shape[0], 1))        
            for i in range(rst_nodules.shape[0]):
                x1, y1, z1, x2, y2, z2 = rst_nodules[i, 0:6]
                x = int(0.5*(x1+x2))
                y = int(0.5*(y1+y2))
                z = int(0.5*(z1+z2))
                try:
                    cube = volume[z-1:z+1, y-3:y+3, x-3:x+3]
                    mean_HUs[i] = np.mean(cube)
                except:
                    continue            
            keep = np.where(mean_HUs<300)[0]
            rst_nodules = rst_nodules[keep]
       
        return rst_nodules    
   
        
    def convert2feedimageofcandi(self, volume, cfg):
        D, H, W = volume.shape
        feed_volume = np.zeros((D, cfg.CANDI_INPUT_SHAPE[0], cfg.CANDI_INPUT_SHAPE[1], 3), dtype=np.float32)
        for z in range(1, D - 1):
            cur_slice = hu2gray(volume[z - 1:z + 2, ...], WL=-700, WW=1000)
            
            cur_slice = np.transpose(cur_slice, [1, 2, 0])
            #imageio.imsave('candi_patch/'+str(z)+'.png', cur_slice)
            cur_slice = cv2.resize(cur_slice, tuple([cfg.CANDI_INPUT_SHAPE[1], cfg.CANDI_INPUT_SHAPE[0]]),
                                   interpolation=cv2.INTER_CUBIC)
            feed_volume[z, ...] = cur_slice
        return feed_volume
            
    def _record_error(self, error):
        self.last_error = error
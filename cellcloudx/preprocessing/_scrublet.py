try:
    from scrublet import Scrublet
except:
    pass
class ScrubletNew(Scrublet):
    def __init__(self, *args,  threshold_method='Minimum', **kwargs):
        # super(ScrubletNew, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.threshold_method = threshold_method

    def call_doublets(self, threshold=None, verbose=True):
        '''
        #bimodal histogram python threshold
        #https://datascience.stackexchange.com/questions/20397/how-to-model-a-bimodal-distribution-of-target-variable
        #https://stackoverflow.com/questions/35990467/fit-mixture-of-two-gaussian-normal-distributions-to-a-histogram-from-one-set-of
        #https://scikit-image.org/docs/0.13.x/api/skimage.filters.html#skimage.filters.threshold_local
        #https://theailearner.com/2019/07/19/balanced-histogram-thresholding/
        #https://stackoverflow.com/questions/42149979/determining-a-threshold-value-for-a-bimodal-distribution-via-kmeans-clustering/42150214
        '''
        threshold_method = self.threshold_method
        if threshold is None:
            # automatic threshold detection
            # http://scikit-image.org/docs/dev/api/skimage.filters.html
            from skimage import filters
            import collections
            methods = collections.OrderedDict({
                        'Isodata': filters.threshold_isodata,
                        'Li': filters.threshold_li,
                        'Mean': filters.threshold_mean,
                        'Minimum': filters.threshold_minimum,
                        'Otsu': filters.threshold_otsu,
                        'Triangle': filters.threshold_triangle,
                        'Yen': filters.threshold_yen})
            try:
                for _i in methods.keys():
                    _t = methods[_i](self.doublet_scores_sim_)
                    print('Automaticall threshold of %s: %.4f'%(_i, _t))
                threshold = methods[threshold_method](self.doublet_scores_sim_)
                if verbose:
                    print("Automatically set threshold with %s at doublet score = %.4f"%(threshold_method, threshold))
            except:
                self.predicted_doublets_ = None
                if verbose:
                    print("Warning: failed to automatically identify doublet score threshold. Run `call_doublets` with user-specified threshold.")
                return self.predicted_doublets_

        Ld_obs = self.doublet_scores_obs_
        Ld_sim = self.doublet_scores_sim_
        se_obs = self.doublet_errors_obs_
        Z = (Ld_obs - threshold) / se_obs
        self.predicted_doublets_ = Ld_obs > threshold
        self.z_scores_ = Z
        self.threshold_ = threshold
        self.detected_doublet_rate_ = (Ld_obs>threshold).sum() / float(len(Ld_obs))
        self.detectable_doublet_fraction_ = (Ld_sim>threshold).sum() / float(len(Ld_sim))
        self.overall_doublet_rate_ = self.detected_doublet_rate_ / self.detectable_doublet_fraction_

        if verbose:
            print('Detected doublet rate = {:.1f}%'.format(100*self.detected_doublet_rate_))
            print('Estimated detectable doublet fraction = {:.1f}%'.format(100*self.detectable_doublet_fraction_))
            print('Overall doublet rate:')
            print('\tExpected   = {:.1f}%'.format(100*self.expected_doublet_rate))
            print('\tEstimated  = {:.1f}%'.format(100*self.overall_doublet_rate_))
        return self.predicted_doublets_
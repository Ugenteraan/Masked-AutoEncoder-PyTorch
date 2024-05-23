'''Module to visualize the prediction done by the decoder.
'''


class VisualizePrediction:

	def __init__(self, 
				 visualize_batch_size=6, 
				 image_size=224,
				 patch_size=16,
				 fig_savepath='./figures/',
				 num_figs=10):


		self.image_size = image_size 
		self.visualize_batch_size = visualize_batch_size
		self.patch_size = patch_size
		self.fig_savepath = fig_savepath
		self.num_figs = num_figs
		




	def plot(self,
		     pred,
		     target,
		     epoch_idx):

		'''Plots both the target and the prediction from the decoder side by side.
		'''

		fig, axes = plt.subplots(nrows=self.visualize_batch_size, ncols=2)

		for idx in range(pred.size(0)):

			predi



import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Plotly for 3D interactive plots
from PIL import Image             # PIL for image manipulation
import os
import imageio                    # Imageio for reading and writing image data
from IPython.display import display
from IPython.display import display, Image
from sklearn.gaussian_process import GaussianProcessRegressor # Scikit-learn for Gaussian process regression
from sklearn.gaussian_process.kernels import Matern           # Scikit-learn for Matern kernel
np.random.seed(42)                                            # Set random seed for reproducibility

from sklearn.neural_network import MLPClassifier   # Classification problem
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class BlackBox:
    def __init__(self):
    
        #Branin parameters
        self.a = 1.0
        self.b = 5.1 / (4.0 * (np.pi ** 2))
        self.c = 5.0 / np.pi
        self.r = 6.0
        self.s = 10.0
        self.t = 1 / (8.0 * np.pi)

        
    @staticmethod        
    def simple_func(x: np.ndarray)-> np.ndarray:
        y = np.sin(x) + np.cos(2*x)
        return y


    @staticmethod
    def branin(x1: np.ndarray,
               x2: np.ndarray,
                a: float, 
                b: float, 
                c: float, 
                r: float, 
                s: float, 
                t: float) -> np.ndarray:
        Hb = np.zeros_like(x1, dtype=float)
        Hb = a * (x2 - b * (x1 ** 2) + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return Hb


        return y_values
        
        
        
# Acquisition function
def prob_i(x, gp_model, best_y):
    """_summary_

    Args:
        x (ndarray shape [n*n, ndims]): grid of values in the range where we evaluate the blackbox function
        gp_model (GaussianProcessRegressor): it's used for fitting Gaussian process regression models to data
        best_y (float): best estimation of max[f(x)] at current step

    Returns:
        pi (ndarray shape [n, ndims]): probability of improvement for each x in the grid
    """    
    if len(x.shape)==1 or x.shape[1]==1:
        y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    else:
        y_pred, y_std = gp_model.predict(x, return_std=True)
    z = (y_pred - best_y) / y_std
    pi = norm.cdf(z)
    return pi

def expected_i(x, gp_model, best_y):
    """_summary_

    Args:
        x (ndarray shape [n*n, ndims]): grid of values in the range where we evaluate the blackbox function
        gp_model (GaussianProcessRegressor): it's used for fitting Gaussian process regression models to data
        best_y (float): best estimation of max[f(x)] at current step

    Returns:
        ei (ndarray shape [n, ndims]): expected improvement for each x in the grid
    """    
    if len(x.shape)==1 or x.shape[1]==1:
        y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    else:
        y_pred, y_std = gp_model.predict(x, return_std=True)
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei
    
    

def smc(x,y,N,k_dim,T, var):
    '''Parameters:
    
    - x = sample of x
    - y = sample of y
    - N = number of samples for smc
    - k_dim = number of hyperparameters
    - T = 
    - var = T/100
    
    Output: theta_best = optimized hyperparameters'''
    
    #inizialization
    theta = np.zeros((N,k_dim,T))
    theta[:,:,0] = np.full((N,k_dim), 10) #to avoid negative theta we start far from 0
    w = np.zeros((N,T))
    
    
    for t in range(1, T):
        for i in range(N):
            for j in range(k_dim):
                theta[i,j,t] = np.random.normal(loc=theta[i,j,t-1], scale=var[j])
            
            
            #compute weights (gp needs to be computed with each set of hyperpars)
            if k_dim==2:
                kernel = (theta[i,0,t]**2) * Matern(length_scale=theta[i,1,t], nu=1.5)
                gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-5)
                gp.fit(x.reshape(-1, 1),y)
            elif k_dim==3:
                kernel = (theta[i,0,t]**2) * Matern(length_scale=[theta[i,1,t],theta[i,2,t]], nu=1.5)
                gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-5)
                gp.fit(x,y)
                
            w[i,t] = np.exp(gp.log_marginal_likelihood()*1e-3) #scaled to avoid underflow; will be normalized, so no worry

        #normalize weights
        w[:,t]/=np.sum(w[:,t])
        
        '''start resampling'''
        #print(w[:,t])
        #resample with replacement:
        for i in range(N):
            index = np.random.choice(N, size=1, p=w[:,t])
            theta[i,:,t] = theta[index,:,t]
        
    theta_best = np.mean(theta[:,:,T-1], axis=0)
    return theta_best
    
    
    
# Plot functions

def plot_1D(matriX, my_blackbox, improv, y_pred, y_std, sample_x, sample_y, new_x, new_y,x_min, x_max, y_min, y_max):
    '''
    Args:
	matriX (ndarray of shape [n*n, ndims]): range of values in the x-axis where the functions will be plotted
	my_blackbox (class): contains the original black box function
	improv (array n): output of the acquisiton function
	y_pred (ndarray of lenght n): predicted values obtained from the Gaussian Process model
	y_std(ndarray of lenght n): standard deviation associated with the predictions obtained from the Gaussian Process model
	sample_x (ndarray of lenght n_sample) and sample_y (ndarray of lenght n_sample): hold the coordinates of the previously sampled points used to train the surrogate model
    	new_x, new_y (float): the new points added to the sampling 
    	
    '''

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(matriX, my_blackbox.simple_func(matriX), color='orange', label='Black Box Function', linewidth=2, alpha=0.7)
    ax1.fill_between(matriX, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
    ax1.plot(matriX, y_pred, color='blue', label='Gaussian Process', alpha=0.7, linewidth=2)
    ax1.scatter(sample_x, sample_y, color='red', label='Previous Points')
    ax1.scatter(new_x, new_y, color='green', label='New Points')
    ax1.grid('--', linewidth = 0.3, color = 'grey')
     
    # Set the fixed limits for the plot
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    #ax2.plot(matriX, my_blackbox.simple_func(matriX), color='orange', label='Black Box Function', linewidth=2, alpha=0.7)
    ax2.plot(matriX, improv, color='red', linestyle='dashed', label='Acquisition Function', alpha=0.8)
    ax2.grid('--', linewidth = 0.3, color = 'grey')
    #ax2.set_ylim(-0.01,0.1)
    ax2.axvline(x=new_x, color='green', linewidth=1.5)  
    
    ax1.legend()
    plt.tight_layout()
 

def make_gif(folder_path, frames, gif_name, duration):
    """
    Takes a folder containing frames and builds a GIF animation.

    Args:
        folder_path (str): The path to the folder containing the image frames.
        frames (list[str]): A list of strings representing the filenames of the frames
                            in the specified order for the animation.
        gif_name (str): the name of the final gif.

    Returns:
        None (creates a GIF file)
    """

    gif_path = os.path.join(folder_path, gif_name)

    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for frame in frames:
            image = imageio.imread(os.path.join(folder_path, frame))
            writer.append_data(image)

# optimization algorithm for branin and for the simple function
def optimize_d12(x_grid,x,y,my_blackbox: BlackBox,optimizer: str,num_iterations: int, acquisition, N, T, var,  folder_path:str, gif_name:str,gif_dur,x1=None,x2=None):

    """_summary_

    Args:
        x_grid (_type_): grid for x locations (1d or 2d)
        x : sample of x points
        y (_type_): observed sample values
        x1 (_type_): _description_
        x2 (_type_): _description_
        num_iterations (_type_): _description_
        acquisition (_type_): _description_
        N (_type_): _description_
        T (_type_): _description_
        var (_type_): _description_
        k_dim (_type_): _description_
        folder_path: folder to save images and gifs in
        gif_name: name of 1d gif or 2d plot
        gif_dur: length of each frame in 1d gif
    """    

    y_min = min(y) - 1
    y_max = max(y) + 1
    x_min = min(x) - 1
    x_max = max(x) + 1

    #dimension of parameters vector
    d = x.reshape(-1,1).shape[1]
    theta_best = [1.0]*(d+1) #initial hyperpars of kernel
    kernel = theta_best[0]**2 * Matern(length_scale=theta_best[1:], nu=1.5)
    
    #set plot for 1d gif:
    if (d==1):
        frames = []
        plt.figure(figsize=(10, 6))
    
    if (optimizer=="fmin_l_bfgs_b"):
        gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer= "fmin_l_bfgs_b")
    


    for i in range(num_iterations):
        if (i%10==0):
            print('Iteration number : ', i)
        
        if (optimizer=="smc"):
        
            #Manually update kernel hyperparameters
            gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer= None)
            
        # Fit the Gaussian process model to the sampled points
        gp_model.fit(x.reshape(-1,d),y)
    
        # Determine the point with the highest observed function value
        best_idx = np.argmax(y)
        best_x = x[best_idx]
        best_y = y[best_idx]
    
        # Generate the acquisition function using the Gaussian process model
        y_pred, y_std = gp_model.predict(x_grid.reshape(-1,d), return_std=True)
                                         
        if acquisition==0:
            improv = expected_i(x_grid.reshape(-1, d),gp_model,best_y)
        else:
            improv = prob_i(x_grid.reshape(-1, d),gp_model,best_y)
            
        if i < num_iterations - 1:
            new_x = x_grid[np.argmax(improv)].reshape(-1,d)
            # Select the next point based on acquisition function
            if d==1:
                new_y = my_blackbox.simple_func(new_x[0])
                x = np.append(x, new_x)
            if d==2:
                new_y= my_blackbox.branin(x1,
                                          x2,
                                          my_blackbox.a,
                                          my_blackbox.b,
                                          my_blackbox.c,
                                          my_blackbox.r,
                                          my_blackbox.s,
                                          my_blackbox.t)
                x = np.concatenate((x, new_x))
            y = np.append(y, new_y)
		
        if (optimizer=="smc"):
        #Optimize hyperparameters with smc
            theta_best = smc(x,y,N,d+1,T,var)
        
        # Save frame for 1d gif
        if d==1:
            
            # Plot the black box function, surrogate function, previous points, and new points
            plot_1D(x_grid, my_blackbox, improv, y_pred, y_std, x, y, new_x, new_y, x_min, x_max, y_min, y_max)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"Iteration #{i+1}")
            plt.legend()
            plt.grid()

            filename = f"frame_{i}.png"

            plt.savefig(os.path.join(folder_path, filename))
            frames.append(filename)
            plt.clf()  # Clear current figure
    


    if d==1:
    # Create the GIF using the frames saved in the specified folder
        make_gif(folder_path, frames, gif_name, gif_dur)
        # Remove the saved frames
        for frame_file in frames:
            os.remove(os.path.join(folder_path, frame_file))
    if d==2:
        # Final plot
        plot_3d_surface_variance(x1,x2, y_pred,y_std, folder_path,name)


        
    print('Optimized theta: ', gp_model.kernel_)
    return(x,y)


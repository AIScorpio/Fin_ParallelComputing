import numpy as np
import pyopencl as cl
from .utils import openCLEnv
import time

import warnings

# Black-Scholes
from scipy.stats import norm

def BlackScholes(S0, K, r, sigma, T, opttype='P'):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    put_price = call_price - S0 + np.exp(-r*T)*K
    
    if opttype == 'C':
        # price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
        price = call_price
    elif opttype == 'P':
        # price = np.exp(-r*T)*K*norm.cdf(-d2) - S0*norm.cdf(-d1) 
        price = put_price
    return price

def BlackScholes_matrix(St, K, r, sigma, T, nPeriod, opttype='P'):
    BS = np.zeros_like(St, dtype=np.float32)
    dt = T / nPeriod

    for t in range(nPeriod):
        new_T = dt * (nPeriod - t)
        BS[:,t] = BlackScholes(St[:,t], K, r, sigma, new_T, 'P')
    return BS

class MonteCarloBase:
    # built-in seeds
    __seed = 1001
    # __seed = np.nan
    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype):
        # init simulation parameters
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.nPath = nPath
        self.nPeriod = nPeriod
        self.K = K
        self.opttype = opttype
        self.opt = None
        match self.opttype:
            case 'C':
                self.opt = -1
            case 'P':
                self.opt = 1
        self.dt = self.T / self.nPeriod

        # generate St simulation
        self.Z = self.__getZ()
        self.St = self.__getSt()
        self.BS = self.__getBlackScholes_matrix()
        
    @classmethod
    def getSeed(cls):
        return cls.__seed
    
    @classmethod
    def setSeed(cls, seed):
        cls.__seed = seed
        # self.Z = self.__getZ()
        # self.St = self.__getSt()
        return
    
    def __getZ(self):
        if self.__seed is np.nan:
            rng = np.random.default_rng()  
        else:
            rng = np.random.default_rng(seed=self.__seed)  
        Z = rng.normal(size=(self.nPath, self.nPeriod)).astype(np.float32)  
        return Z
    
    def __getSt(self):
        # pre-compute Geometric Brownian Motion parameters
        nudt = (self.r - 0.5 * self.sigma**2) * self.dt       # drift component
        volsdt = self.sigma * np.sqrt(self.dt)                # diffusion component
        lnS0 = np.log(self.S0)                           # using log normally distributed feature
        
        # log price approach
        delta_lnSt = nudt + volsdt * self.Z    # nPeriod by nPath
        lnSt = lnS0 + np.cumsum(delta_lnSt, axis=1) 
        # lnSt = np.concatenate( (np.full(shape=(1, nPath), fill_value=lnS0), lnSt))
        St = np.exp(lnSt).astype(np.float32)
        
        return St
    
    def __getBlackScholes_matrix(self):

        BS = np.zeros_like(self.St, dtype=np.float32)
        dt = self.T / self.nPeriod
        
        for t in range(self.nPeriod):
            new_T = dt * (self.nPeriod - t)
            BS[:,t] = BlackScholes(self.St[:,t], self.K, self.r, self.sigma, new_T, self.opttype)
        return BS
    
    # get MC St payoffs
    def getPayoffs(self):
        # immediate exercise payoffs for each path and time step
        payoffs = np.maximum(0, (self.K - self.St) * self.opt)
        
        return payoffs

    
# Monte Carlo Simulation for American Put Option Pricing
class hybridMonteCarlo(MonteCarloBase):
    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish):
        super().__init__(S0, r, sigma, T, nPath, nPeriod, K, opttype)
        # initialize parameters
        self.nFish = nFish
        
        # init buffer for Z and St for Pso 
        self.Z_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Z)
        # self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.St)

        # # Initialize shared arrays for Pso child classes
        nDim = self.nPeriod
        rng = np.random.default_rng(seed=52)
        # rng = np.random.default_rng()

        # self.pos_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * 100.0    #self.S0
        # self.vel_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * 5.0      #np.abs(self.S0 - self.K)
        
        high = max(self.S0, self.K)
        self.pos_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * high
        self.vel_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * high * 0.05

        self.r1 = rng.uniform(size=(nDim, nFish)).astype(np.float32)
        self.r2 = rng.uniform(size=(nDim, nFish)).astype(np.float32)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    
    # Monte Carlo European option - CPU
    def getEuroOption_np(self):
        start = time.perf_counter()
        assert (self.St.shape[0] == (np.exp(-self.r*self.T) * np.maximum(0, (self.K - self.St[:, -1]) * self.opt) ).shape[0])
        C_hat_Euro = (np.exp(-self.r*self.T) * np.maximum(0, (self.K - self.St[:, -1]) * self.opt) ).sum() / self.nPath
    
        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo Numpy European price: {C_hat_Euro} - {elapse} ms")
        return C_hat_Euro, elapse

    # Monte Carlo European option - GPU
    def getEuroOption_cl(self):        
        warnings.warn(
            "getEuroOption_cl is NOT performing, and will be removed in future versions, use `getEuroOption_cl_optimize` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        start = time.perf_counter()  
        prog_EuroOpt = cl.Program(openCLEnv.context, open("./models/kernels/mc/knl_source_mc_getEuroOption.c").read()%(self.nPath, self.nPeriod)).build()
        knl_getEuroOption = cl.Kernel(prog_EuroOpt, 'getEuroOption')

        # prepare result array, length of nPath for kernel threads
        payoffs = np.empty(self.nPath, dtype=np.float32)  # length of npath
        payoffs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=payoffs.nbytes)
            
        # current
        knl_getEuroOption.set_args(self.Z_d, np.float32(self.S0), np.float32(self.K), 
                                  np.float32(self.r), np.float32(self.sigma), 
                                  np.float32(self.T), np.int8(self.opt), payoffs_d)
        
        # run kernel
        global_size = (self.nPath, )
        local_size = None
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_getEuroOption, global_size, local_size)
        cl.enqueue_copy(openCLEnv.queue, payoffs, payoffs_d, wait_for=[evt])
        openCLEnv.queue.finish()    # <------- sychrnozation
        
        C_hat_Euro = payoffs.sum() / self.nPath
        
        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo {openCLEnv.deviceName} European price: {C_hat_Euro} - {elapse} ms")
        return C_hat_Euro

    def getEuroOption_cl_optimized(self):      
        start = time.perf_counter()        
        kernel_src = open("./models/kernels/mc/knl_source_mc_getEuroOption.c").read()
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros"]
        # prog_EuroOpt = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_mc_getEuroOption.c").read()%(self.nPath, self.nPeriod)).build()
        prog_EuroOpt = cl.Program(openCLEnv.context, kernel_src %(self.nPath, self.nPeriod)).build(options=build_options)
        knl_getEuroOption = cl.Kernel(prog_EuroOpt, 'getEuroOption_optimized')

        # prepare result array, length of nPath for kernel threads
        payoffs = np.empty(self.nPath, dtype=np.float32)  # length of npath
        payoffs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=payoffs.nbytes)
 
        # optimized
        # lnS0 = np.log(self.S0)
        # exp_neg_rT = np.exp(- self.r * self.T)
        knl_getEuroOption.set_args(self.Z_d, np.float32(np.log(self.S0)), np.float32(self.K), 
                                  np.float32(self.r), np.float32(self.sigma), 
                                  np.float32(self.T), np.int8(self.opt), np.float32(np.exp(- self.r * self.T)), payoffs_d)
        
        # run kernel
        global_size = (self.nPath, )
        local_size = None
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_getEuroOption, global_size, local_size)
        cl.enqueue_copy(openCLEnv.queue, payoffs, payoffs_d, wait_for=[evt])
        openCLEnv.queue.finish()    # <------- sychrnozation
        
        C_hat_Euro = payoffs.sum() / self.nPath
        
        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo {openCLEnv.deviceName} European price: {C_hat_Euro} - {elapse} ms")
        return C_hat_Euro, elapse

    def getEuroOption_cl_optimize_reductionSum(self):    
        CEILING = 65536
        if (self.nPath > CEILING):
            warnings.warn(
                f"getEuroOption_cl_optimize_reductionSum ONLY works no exceeding {CEILING} paths, use `getEuroOption_cl_optimize` instead.",
                UserWarning,
                stacklevel=2
            )   
        start = time.perf_counter()        
        prog_EuroOpt = cl.Program(openCLEnv.context, open("./models/kernels/mc/knl_source_mc_getEuroOption.c").read()%(self.nPath, self.nPeriod)).build()
        knl_getEuroOption_sum1 = cl.Kernel(prog_EuroOpt, 'getEuroOption_optimized_sum1')
        knl_getEuroOption_sum2 = cl.Kernel(prog_EuroOpt, 'getEuroOption_optimized_sum2')

        # prepare result array, length of nPath for kernel threads
        # Calculate padded path count (multiple of 4 for float4)
        float32_bytes = np.dtype(np.float32).itemsize     # sizeof(float): 4 

        # Get workgroup size for local memory
        max_wg_size = openCLEnv.device.max_work_group_size
        preferred_wg_size = 256  # Optimal for most GPUs
        WORKGROUP_SIZE = min(preferred_wg_size, max_wg_size)

        # Local memory for reduction
        # local_mem = cl.LocalMemory(WORKGROUP_SIZE * float32_bytes)

        ## First reduction
        # optimized
        # lnS0 = np.log(self.S0)
        # exp_neg_rT = np.exp(- self.r * self.T)
        global_size = (self.nPath, )
        local_size = (min(global_size[0], WORKGROUP_SIZE), )
        n_Groups = (global_size[0] + local_size[0] - 1) // local_size[0]
        C_hats = np.empty(n_Groups, dtype=np.float32)
        C_hats_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=C_hats.nbytes)     # sizeof(float): 4
        # print(f'>> 1 reduction: {self.nPath} threads, {WORKGROUP_SIZE} per group, {n_Groups} groups')

        knl_getEuroOption_sum1.set_args(self.Z_d, np.float32(np.log(self.S0)), np.float32(self.K), 
                                  np.float32(self.r), np.float32(self.sigma), 
                                  np.float32(self.T), np.int8(self.opt), np.float32(np.exp(- self.r * self.T)), 
                                  cl.LocalMemory(WORKGROUP_SIZE * float32_bytes), C_hats_d)

        # Second/Final reduction        
        global_size2 = (n_Groups, )
        local_size2 = (min(n_Groups, WORKGROUP_SIZE), )
        # print(f'>> {global_size2[0] // local_size2[0]}, { (n_Groups + local_size2[0] - 1) // local_size2[0]}')

        final_result = np.empty(1, dtype=np.float32)
        # n_Groups2 = (global_size2[0] + local_size2[0] - 1) // local_size2[0]
        # final_result = np.empty(n_Groups2, dtype=np.float32)         # if too many paths exceeding e.g. 256^2, there will be (global_size2[0] // local_size2[0]) partial sums
        final_result_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=final_result.nbytes) 
        # print(f'>> 2 reduction: {n_Groups} threads, {local_size2[0]} per group, {n_Groups2} groups')

        knl_getEuroOption_sum2.set_args(np.int8(n_Groups), C_hats_d, cl.LocalMemory(local_size2[0] * float32_bytes), final_result_d)
        
        # run kernel
        evt1 = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_getEuroOption_sum1, global_size, local_size)
        # cl.enqueue_copy(openCLEnv.queue, C_hats, C_hats_d, wait_for=[evt1])    # if only first reduction
        evt2 = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_getEuroOption_sum2, global_size2, local_size2, wait_for=[evt1])
        cl.enqueue_copy(openCLEnv.queue, final_result, final_result_d, wait_for=[evt2])
        openCLEnv.queue.finish()    # <------- sychrnozation
        
        # C_hat_Euro = C_hats.sum() / self.nPath      # if only first reduction
        C_hat_Euro = final_result.sum() / self.nPath

        elapse = (time.perf_counter()- start) * 1e3
        print(f"MonteCarlo {openCLEnv.deviceName}-reductionSum European price: {C_hat_Euro} - {elapse} ms")
        return C_hat_Euro

    # # Monte Carlo pso American option - CPU: take one particle each time and loop thru PSO
    # def costPsoAmerOption_np(self, in_particle):
    #     # # udpated on 6 Apr. 2025 
    #     # 1. for unified Z, St shape as nPath by nPeriod, synced and shared by PSO and Longstaff
    #     # 2. No concatenation of spot price
    #     # 3. handle index of time period, spot price at time zero (present), St from time 1 to T

    #     # get the boundary index where early cross (particle period > St period), as if an early exercise judgement by this fish/particle
    #     boundaryIdx = np.argmax(self.St < in_particle[None, :], axis=1)   # [0, 1] as of true or false of early cross

    #     # if no, set boundary index to last time period, meaning no early exercise suggested for that path
    #     boundaryIdx[boundaryIdx==0] = self.nPeriod - 1    # to handle time T index for boundary index to match St time wise dimension (i.e. indexing from zero)
        
    #     # determine exercise prices by getting the early cross St_ij on path i and period j
    #     exerciseSt = self.St[np.arange(len(boundaryIdx)), boundaryIdx]    # len of boundaryIdx is nPath
        
    #     # discounted back to time zero, hence boundaryIdx+1
    #     searchCost = (np.exp(-self.r * (boundaryIdx+1) * self.dt) * np.maximum(0, (self.K - exerciseSt)*self.opt) ).sum() / self.nPath

    #     return searchCost

    # # Monte Carlo pso American option - GPU: take the whole PSO and process once
    # def costPsoAmerOption_cl(self, pso_buffer, costs_buffer):
        
    #     self.knl_psoAmerOption_gb.set_args(self.St_d, pso_buffer, costs_buffer, 
    #                                        self.boundary_idx_d, self.exercise_d, 
    #                                        np.float32(self.r), np.float32(self.T), np.float32(self.K), np.int8(self.opt))

    #     # execute kernel
    #     global_size = (self.nFish, )
    #     local_size = None
    #     cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_psoAmerOption_gb, global_size, local_size).wait()
    #     openCLEnv.queue.finish()    # <------- sychrnozation

    #     return 

    def cleanUp(self):
        self.Z_d.release()
        # self.St_d.release()
        # self.boundary_idx_d.release()
        # self.exercise_d.release()
        return 



def main():
    S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 10, 3, 100.0, 'P', 500
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
    print(mc.St)

if __name__ == "__main__":
    main()

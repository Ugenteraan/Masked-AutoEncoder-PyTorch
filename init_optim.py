'''PyTorch Optimizer with learning rate and weight decay schedulers.
'''

import math
import torch
import torch.optim as optim


class InitOptimWithSGDR:
    '''Initialize an Optimizer with Stochastic Gradient Descent with Restarts (SGDR a.k.a Cosine Annealing) with Linear WarmUp strategy and weight decay (L2 Regularization).
       The weight decay rate will also be calculated using the cosine annealing strategy.
       The implementation here will use more frequent restarts, then after the ```epoch_idx_to_increase_restarts```, the restarts will be much infrequent.
    '''


    def __init__(self, 
                 autoencoder_model,
                 cosine_upper_bound_lr, 
                 cosine_lower_bound_lr, 
                 warmup_start_lr, 
                 warmup_steps,
                 initial_num_steps_to_restart_lr,
                 final_num_steps_to_restart_lr,
                 epoch_idx_to_increase_restarts,
                 cosine_upper_bound_wd,
                 cosine_lower_bound_wd,
                 upper_bound_lr_decay,
                 logger=None):

        self.cosine_upper_bound_lr = cosine_upper_bound_lr
        self.cosine_lower_bound_lr = cosine_lower_bound_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_steps = warmup_steps
        self.num_steps_to_restart_lr = initial_num_steps_to_restart_lr #the LHS value will get replaced with ```final_num_steps_to_restart_lr``` later.
        self.final_num_steps_to_restart_lr = final_num_steps_to_restart_lr
        self.epoch_idx_to_increase_restarts = epoch_idx_to_increase_restarts
        self.step_counter = 0
        self.cosine_upper_bound_wd = cosine_upper_bound_wd
        self.cosine_lower_bound_wd = cosine_lower_bound_wd
        self.upper_bound_lr_decay = upper_bound_lr_decay
        self.transition_flag = False #boolean used to keep track of the epoch has surpassed the ```epoch_idx_to_increase_restarts```.


        self.logger = logger

        # #we want to apply weight decay only to the weights. Not the biases. Therefore, we'll create two separate groups of params for the model.
        param_group = [
                    #the checks are for the bias (name variable) and it's shape (p variable) to exclude/include them.
                    { 
                        'params': (p for n, p in autoencoder_model.named_parameters() if ('bias' not in n) and (len(p.size()) != 1))
                    },
                    #The weight decay is set to 0 and there's a bool var to ensure the weight decay update later doesn't affect them.
                    {
                        'params': (p for n, p in autoencoder_model.named_parameters() if ('bias' in n) and (len(p.size()) == 1)),
                        'WD_exclude': True,
                        'weight_decay':0
                    },
                ]


        self.optimizer = torch.optim.AdamW(param_group)
        
        assert cosine_upper_bound_lr >= cosine_lower_bound_lr, "Upper bound for LR needs to be bigger or equal to the lower bound"
        assert cosine_upper_bound_wd >= cosine_lower_bound_wd, "Upper bound for weight decay needs to be bigger or equal to the lower bound"

    def get_optimizer(self):
        '''Returns the optimizer.
        '''
        return self.optimizer

    def cosine_annealing(self, start_value, end_value, fraction_term):
        '''To calculate the new learning rate and the weight decay rate using the cosine annealing strategy.
        '''
        res = start_value + 0.5 * (end_value - start_value) * (1. + math.cos(math.pi * fraction_term))
        return res

    


    def step(self, epoch_idx=None):
        '''Must be executed at every iteration step (not epoch step).
        '''

        self.step_counter += 1

        #we're gonna need to write 2 piece of logics. 1 for the warm up period. And 1 for after the warm up period.
        if self.step_counter <= self.warmup_steps:
            fraction_term = (self.cosine_upper_bound_lr - self.warmup_start_lr)/(self.warmup_steps) #we don't need the -1 in the denominator since we're starting the steps from 1 not 0.
            new_lr = self.warmup_start_lr + self.step_counter * fraction_term

        else:
            #cosine annealing after the warmup.
            fraction_term = float(self.step_counter - self.warmup_steps) / float(max(1, self.num_steps_to_restart_lr))
            new_lr = max(self.cosine_lower_bound_lr, self.cosine_annealing(start_value=self.cosine_lower_bound_lr,
                                                                           end_value=self.cosine_upper_bound_lr,
                                                                           fraction_term=fraction_term))

            



            #it's >= in case a trained model is loaded after the said epoch value.
            if epoch_idx >= self.epoch_idx_to_increase_restarts and not self.transition_flag: #transition to the bigger num of steps to restart.

                self.num_steps_to_restart_lr = self.final_num_steps_to_restart_lr
                self.cosine_upper_bound_lr = self.cosine_upper_bound_lr * self.upper_bound_lr_decay

                self.transition_flag = True #this block of code will never be executed again.


            #once the learning rate reaches the lower bound, restart the learning rate back to the upper bound value.
            if new_lr == self.cosine_lower_bound_lr:
                self.step_counter += self.num_steps_to_restart_lr

                
        
        #calculate the weight decay rate. There is no warmup period for decay rate and we will be using the same num of steps we used for lr for the restart.
        fraction_term = self.step_counter / self.num_steps_to_restart_lr
        new_wd = self.cosine_annealing(start_value=self.cosine_lower_bound_wd,
                                       end_value=self.cosine_upper_bound_wd,
                                       fraction_term=fraction_term)
        
        #update the optimizer with the new learning rate and the new weight decay rate.
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
            
            #check for the weight decay variable so that we don't modify the bias parameter.
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
            

        #we are returning the lr and wd for logging purposes only. The optimizer is already updated with the new values.
        return new_lr, new_wd



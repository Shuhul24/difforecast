import os

def get_deepspeed_config(args):
        config_params = {
            'train_batch_size': int(os.environ['WORLD_SIZE']) * args.batch_size,
        }
        config_params['flops_profiler'] = {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }
        # config_params['zero_optimization'] ={
        #     'stage': 1,
        # }
        
        # ZeRO stage 0 is used to avoid the count_used_parameters_in_backward
        # assertion that requires internal PyTorch APIs unavailable in this build.
        # With a single GPU, ZeRO stages 1/2 provide no memory benefit from
        # optimizer-state partitioning, so stage 0 is equivalent in practice.
        config_params["zero_optimization"] = {
            "stage": 0,
        }


        # config_params["train_micro_batch_size_per_gpu"] = int(args.batch_size),
        # # config_params["train_batch_size"]="auto",
        # config_params["gradient_accumulation_steps"]="auto",
        # config_params['zero_optimization'] = {
        #     "stage": 3,
        #     "overlap_comm": True, 
        #     "contiguous_gradients": True, 
        #     "sub_group_size": 1e9, 
        #     "reduce_bucket_size": "auto", 
        #     "stage3_prefetch_bucket_size": "auto",
        #     "stage3_param_persistence_threshold": "auto",
        #     "stage3_max_live_parameters": 1e9,
        #     "stage3_max_reuse_distance": 1e9,
        #     "stage3_gather_16bit_weights_on_model_save": True

        # }
        config_params['bf16'] = {
            "enabled": True,
        }
        config_params['zero_allow_untested_optimizer'] = True

        return config_params
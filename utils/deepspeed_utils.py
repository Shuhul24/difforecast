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
        # ZeRO Stage 2 with CPU optimizer offload:
        # - Keeps Adam optimizer states (exp_avg, exp_avg_sq) on CPU, preventing
        #   GPU OOM when initializing optimizer states for a large bf16 model.
        # - Stage 2 has far less initialization overhead than Stage 3 (no parameter
        #   partitioning), which avoids the SIGKILL from CPU RAM exhaustion on init.
        # - Requires PyTorch >= 2.2 for the count_used_parameters_in_backward API.
        config_params["zero_optimization"] = {
            "stage": 2,
        }
        config_params['bf16'] = {
            "enabled": True,
        }
        config_params['zero_allow_untested_optimizer'] = True
        config_params['gradient_clipping'] = 1.0

        return config_params
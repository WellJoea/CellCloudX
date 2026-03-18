import inspect
def clean_cache(*array, iter=3, reset=None):
    if len(array): #TODO
        for var_name in array:
            caller_globals = inspect.currentframe().f_back.f_globals
            if var_name in caller_globals:
                del caller_globals[var_name]
            else:
                #raise NameError(f"Variable '{var_name}' not found in global scope")
                pass
    import gc
    i = 0
    while (gc.collect()) and (i <iter):
        gc.collect()
        i += 1

    try:
        import torch as th
        th.cuda.empty_cache()
        th.cuda.empty_cache()
    except:
        pass

    if reset is not None:
        try:
            import torch as th
            th.cuda.reset_max_memory_allocated(reset)
        except:
            pass

    import gc
    i = 0
    while (gc.collect()) and (i <iter):
        gc.collect()
        i += 1


def get_memory(device):
    if hasattr(device, 'type'):
        device = device.type
    if type(device) == int:
        device = f'cuda:{device}'
    if 'cuda' in str(device):
        try:
            import torch as th
        except:
            return 'torch not found'
        tm = th.cuda.get_device_properties(device).total_memory/(1024**3)
        #am = self.xp.cuda.memory_allocated(device)
        rm = th.cuda.memory_reserved(device)/(1024**3)
        am = tm - rm
        return f'{device} T={tm:.2f};A={am:.2f}GB'
    else:
        # gm = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")/(1024**3)
        import psutil
        tm = psutil.virtual_memory().total/(1024**3)
        am = psutil.virtual_memory().available/(1024**3)
        return f'cpu T={tm:.2f};A={am:.2f}GB'

def get_memory_infor(return_infor=False):
    infors = []
    try:
        import torch as th
        cuda_num = th.cuda.device_count()
        for id in range(cuda_num):
            name = th.cuda.get_device_name(id)
            info = get_memory(id)
            infors.append(f'{name} {info}')
    except:
        return 'torch not found'
    infors.append(get_memory('cpu'))

    if return_infor:
        return infors
    else:
        print('\n'.join(infors))
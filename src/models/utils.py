import pyopencl as cl

class openCLEnv:
    # ready the PyOpenCL environment
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    # print('Initializing opencl context on', cl.device_type.to_string(self.device.type).rpartition(' ')[2])
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    deviceName = 'OpenCL_' + cl.device_type.to_string(device.type).rpartition(' ')[2]
    # print('device name:', self.deviceName)


def checkOpenCL():
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    assert(len(cl.get_platforms())>0)

    for platform in cl.get_platforms():
        print('=' * 60)
        print('Platform - Name: ' + platform.name)
        print('Platform - Vendor: ' + platform.vendor)
        print('Platform - Version: ' + platform.version)
        print('Platform - Profile: ' + platform.profile)

    for device in platform.get_devices():
        print(' ' + '-' * 56)
        print(' Device - Name: ' \
        + device.name)
        print(' Device - Type: ' \
        + cl.device_type.to_string(device.type))
        print(' Device - Max Clock Speed: {0} Mhz'\
        .format(device.max_clock_frequency))
        print(' Device - Compute Units: {0}'\
        .format(device.max_compute_units))
        print(' Device - Global Memory: {0:.0f} GB'\
        .format(device.global_mem_size/ 2**30))
#         print(' Device - Global cache: {0:.0f} KB'\
#         .format(device.global_mem_cache_size/ 2**10))
        print(' Device - Constant Memory: {0:.0f} GB'\
        .format(device.max_constant_buffer_size/ 2**30))
        print(' Device - Local Memory: {0:.0f} KB'\
        .format(device.local_mem_size/ 2**10),  '('+cl.device_local_mem_type.to_string(device.local_mem_type)+')')
    
        
        print(' Device - Max Buffer/Image Size: {0:.0f} GB'\
        .format(device.max_mem_alloc_size/ 2**30))
        print(' Device - Max Work Group Size: {0:.0f}'\
        .format(device.max_work_group_size))
        print(' Device - Float Atomics Supported:', 'cl_khr_global_int32_base_atomics' in device.extensions)
        
        # print(f"preferred_vector_width_float: {device.preferred_vector_width_float}")
        # print(f"native_vector_width_float: {device.native_vector_width_float}")

        print('\n') 



if __name__ == "__main__":
    checkOpenCL()
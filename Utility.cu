#include "Utility.cuh"

GpuConfig& GpuConfig::GetInstance() {
	//Thanks Meyer: check book 
	static GpuConfig _instance;
	return _instance;
}

GpuConfig::GpuConfig() {
	
	int num_of_devices{ 0 };
	auto errata{ cudaGetDeviceCount(&num_of_devices) };

	if (errata == cudaSuccess)
	{
		/// <summary>
		/// We will assume that we are only using one and only one gpu
		/// for this project.
		/// </summary>
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop,0);

		_name=prop.name;
		_id=0;
		_cuda_version=prop.major;
		_total_memory=prop.totalGlobalMem;
		_shared_mem_per_block=prop.sharedMemPerBlock;
		_total_const_mem=prop.totalConstMem;
		_registers_per_block=prop.regsPerBlock;
		_warp_size=prop.warpSize;
		_max_threads_per_block=prop.maxThreadsPerBlock;

		__copy(prop.maxThreadsDim, _max_threads_dim, 3);
		__copy(prop.maxGridSize, _max_grid_size, 3);

		_multiprocessor_count=prop.multiProcessorCount;
	}
}

string	 GpuConfig::Name() const {
	return _name;
}
int_32	 GpuConfig::Id() const {
	return _id;
}
int_32	 GpuConfig::CudaVersion() const {
	return _cuda_version;
}
size_t	 GpuConfig::TotalDeviceMem() const {

	return _total_memory;

}
size_t	 GpuConfig::TotalConstMem()	 const {
	return _total_const_mem;
}
size_t	 GpuConfig::SharedBlockMem() const {
	
	return _shared_mem_per_block;
}
int_32	 GpuConfig::RegistersPerBlock() const {
	return _registers_per_block;
}
int_32	 GpuConfig::WarpSize()	const {
	return _warp_size;
}
int_32	 GpuConfig::MaxThreadPerBlock() const {
	return _max_threads_per_block;
}
const int*    GpuConfig::MaxThreadDim(){
	return _max_threads_dim;
}
const int*    GpuConfig::MaxGridSize(){
	return _max_grid_size;
}
int_32	 GpuConfig::MultiprocessorCount() const {
	return _multiprocessor_count;

}
void	GpuConfig::PrintInfo(){
	cout << "Gpu Name:" << Name() << "\n";
	cout << "Cuda Version:" << CudaVersion() << "\n";
	cout << "Total Global Memory:"<< TotalDeviceMem() << " B  ~"<<GetApproxSize(TotalDeviceMem(),UnitMeasure::GB)<<" GB\n";
	cout << "Total Const  Memory:" << TotalConstMem() << " B  ~"<<GetApproxSize(TotalConstMem(),UnitMeasure::KB)<<" KB\n";
	cout << "Shared Memory Per Block:"  << SharedBlockMem() << " B  ~"<<GetApproxSize(SharedBlockMem() ,UnitMeasure::KB)<<" KB\n";
	cout << "Registers Per Block:" << RegistersPerBlock() << "\n";
	cout << "Warp Size:" << WarpSize() << "\n";
	const auto* tmp = MaxThreadDim();
	cout << "Max Thread Dim: x:" <<tmp[0]<<" y:"<<tmp[1]<<" z:"<<tmp[2] << "\n";
	tmp = MaxGridSize();
    cout << "Max Grid Dim: x:" << tmp[0] << " y:" << tmp[1] << " z:" << tmp[2] << "\n";
	cout << "Multi Processors Count (SM):" << MultiprocessorCount() << "\n";



}

float GpuConfig::extract_size(size_t bytes, const UnitMeasure sz) noexcept {
    //simple function with low decimal precision

    size_t _whole{ bytes / static_cast<size_t>(sz) };   
    bytes -= (static_cast<size_t>(sz) * _whole);
    switch (sz)
    {
    case UnitMeasure::TB:
        //go for gb in digit
        return _whole + (bytes / static_cast<size_t>(UnitMeasure::GB)) * 0.001;
    case UnitMeasure::GB:
        //go for mB in digit
        return _whole + (bytes / static_cast<size_t>(UnitMeasure::MB)) * 0.001;
    case UnitMeasure::MB:
        return _whole + (bytes / static_cast<size_t>(UnitMeasure::KB)) * 0.001;
    case UnitMeasure::KB:
        return _whole + (bytes / static_cast<size_t>(UnitMeasure::B)) * 0.001;
    default:
        //unexpected UnitMeasure size
        return -1;
    } 
}

float GpuConfig::GetApproxSize(size_t mem,const UnitMeasure ms){
	return extract_size(mem,ms);
}


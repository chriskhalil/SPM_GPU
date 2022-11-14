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

		__copy<int,3>(prop.maxThreadsDim, _max_threads_dim, 3);
		__copy<int,3>(prop.maxGridSize, _max_grid_size, 3);

		_multiprocessor_count=prop.multiProcessorCount;
	}
}

const string	 GpuConfig::Name()					     const {
	return _name;
}
const int_32	 GpuConfig::Id()						 const {
	return _id;
}
const int_32	 GpuConfig::CudaVersion()				 const {
	return _cuda_version;
}
const double	 GpuConfig::TotalDeviceMem( UnitMeasure ms = UnitMeasure::B) const {
	if (ms == UnitMeasure::GB) return _total_memory / (1024 * 3);
	if (ms == UnitMeasure::MB) return _total_memory / (1024 * 2);
	if (ms == UnitMeasure::KB) return _total_memory / (1024 * 1);
	return _total_memory;

}
const double	 GpuConfig::TotalConstMem(UnitMeasure ms = UnitMeasure::B)	 const {
	if (ms == UnitMeasure::GB) return _total_const_mem/ (1024 * 3);
	if (ms == UnitMeasure::MB) return _total_const_mem/ (1024 * 2);
	if (ms == UnitMeasure::KB) return _total_const_mem/ (1024 * 1);
	return _total_const_mem;
}
const double	 GpuConfig::SharedBlockMem(UnitMeasure ms = UnitMeasure::B) const {
	if (ms == UnitMeasure::GB) return _shared_mem_per_block / (1024 * 3);
	if (ms == UnitMeasure::MB) return _shared_mem_per_block / (1024 * 2);
	if (ms == UnitMeasure::KB) return _shared_mem_per_block / (1024 * 1);
	return _shared_mem_per_block;
}
const int_32	 GpuConfig::RegistersPerBlock()		 const {
	return _registers_per_block;
}
const int_32	 GpuConfig::WarpSize()				 	 const {
	return _warp_size;
}
const int_32	 GpuConfig::MaxThreadPerBlock()		 const {
	return _max_threads_per_block;
}
const int_32*    GpuConfig::MaxThreadDim()				 const {
	return _max_threads_dim;
}
const int_32*    GpuConfig::MaxGridSize()				 const {
	return _max_grid_size;
}
const int_32	 GpuConfig::MultiprocessorCount()		 const {
	return _multiprocessor_count;

}
const void	GpuConfig::PrintInfo(UnitMeasure ms)						const {
	cout << "Gpu Name:" << Name() << "\n";
	cout << "Cuda Version:" << CudaVersion() << "\n";
	if (ms == UnitMeasure::GB)
		cout << "--------Memory Unit: GB---------\n";
	else if (ms == UnitMeasure::MB)
		cout << "--------Memory Unit: MB---------\n";
	else if (ms == UnitMeasure::KB)
		cout << "--------Memory Unit: KB---------\n";
	else
		cout << "--------Memory Unit:  B---------\n";

	cout << "Total Global Memory:" <<std::setprecision(2) << TotalDeviceMem(ms) << "\n";
	cout << "Total Const  Memory:" << std::setprecision(2) << TotalConstMem(ms) << "\n";
	cout << "Shared Memory Per Block:" << std::setprecision(2) << SharedBlockMem(ms) << "\n\n";
	cout << "Registers Per Block:" << RegistersPerBlock() << "\n";
	cout << "Warp Size:" << WarpSize() << "\n";
	auto tmp = MaxThreadDim();
	cout << "Max Thread Dim: x:" <<tmp[0]<<" y:"<<tmp[1]<<" z:"<<tmp[2] << "\n";
	tmp = MaxGridSize();
    cout << "Max Grid Dim: x:" << tmp[0] << " y:" << tmp[1] << " z:" << tmp[2] << "\n";
	cout << "Multi Processors Count (SM):" << MultiprocessorCount() << "\n";



}

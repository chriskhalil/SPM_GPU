#pragma once

#include <string>
#include <cstdint>
#include <iostream>
#include <iomanip>  

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define int_32 std::int_fast32_t

using std::string;
using std::cout;
enum class UnitMeasure{B,KB,MB,GB};
class GpuConfig final {

	//all mem size is in bytes by defaults
private:
	//default members
	string	_name{""};
	int_32	_id{-1};
	int_32	_cuda_version{0};
	size_t	_total_memory{0};
	size_t	_shared_mem_per_block{0};
	size_t	_total_const_mem{0};
	int_32	_registers_per_block{0};
	int_32	_warp_size{0};
	int_32	_max_threads_per_block{0};
	int	_max_threads_dim[3]{0};
	int	_max_grid_size[3]{0};
	int_32	_multiprocessor_count{0};

private:
	//singleton
	GpuConfig();
	template<typename T,size_t sz>
	void __copy(T (& from)[sz], T (&dest)[sz], size_t num_of_elemets) {
		//simple copy 
		//no need for boundry check in this usecase
		for (size_t i{ 0 }; i < num_of_elemets; ++i)
			dest[i] = from[i];
	}

public:
	const string	 Name()					     const;
	const int_32	 Id()						 const;
	const int_32	 CudaVersion()				 const;
	const double	 TotalDeviceMem(UnitMeasure) const;
	const double	 TotalConstMem(UnitMeasure)	 const;
	const double	 SharedBlockMem(UnitMeasure) const;
	const int_32	 RegistersPerBlock()		 const;
	const int_32	 WarpSize()				 	 const;
	const int_32	 MaxThreadPerBlock()		 const;
	const int_32*    MaxThreadDim()				 const;
	const int_32*    MaxGridSize()				 const;
	const int_32	 MultiprocessorCount()		 const;
	const void		 PrintInfo(UnitMeasure)		 const;

	static GpuConfig& GetInstance();
	GpuConfig(const GpuConfig&) = delete;
	void operator=(const GpuConfig&) = delete;
	~GpuConfig() = default;
};
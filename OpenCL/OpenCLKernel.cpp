/* 
* OpenCL Raytracer
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#ifdef USE_DIRECTX
#include <CL/cl_d3d10_ext.h>
#endif // USE_DIRECTX

#ifdef LOGGING
#include <ETWLoggingModule.h>
#include <ETWResources.h>
#else
#define LOG_INFO( msg ) std::cout << msg << std::endl;
#define LOG_ERROR( msg ) std::cerr << msg << std::endl;
#endif

#include "OpenCLKernel.h"

const long MAX_SOURCE_SIZE = 65535;
const long MAX_DEVICES = 10;

#ifdef USE_DIRECTX
// DirectX
clGetDeviceIDsFromD3D10NV_fn clGetDeviceIDsFromD3D10NV = NULL;
ID3D10Device*           g_pd3dDevice = NULL; // Our rendering device
#endif // USE_DIRECTX

/*
* getErrorDesc
*/
std::string getErrorDesc(int err)
{
	switch (err)
	{
	case CL_SUCCESS                        : return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND               : return "CL_DEVICE_NOT_FOUND";
	case CL_COMPILER_NOT_AVAILABLE         : return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE  : return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES               : return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY             : return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE   : return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP               : return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH          : return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED     : return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE          : return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE                    : return "CL_MAP_FAILURE";

	case CL_INVALID_VALUE                  : return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE            : return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM               : return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE                 : return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT                : return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES       : return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE          : return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR               : return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT             : return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE             : return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER                : return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY                 : return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS          : return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM                : return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE     : return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME            : return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION      : return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL                 : return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX              : return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE              : return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE               : return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS            : return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION         : return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE        : return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE         : return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET          : return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST        : return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_OPERATION              : return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT              : return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE            : return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL              : return "CL_INVALID_MIP_LEVEL";
	default: return "UNKNOWN";
	}
}

/*
* Callback function for clBuildProgram notifications
*/
void pfn_notify(cl_program, void *user_data)
{
	std::stringstream s;
	s << "OpenCL Error (via pfn_notify): " << user_data;
	std::cerr << s.str() << std::endl;
}

/*
* CHECKSTATUS
*/

#define CHECKSTATUS( stmt ) \
{ \
	int __status = stmt; \
	if( __status != CL_SUCCESS ) { \
	std::stringstream __s; \
	__s << "==> " #stmt "\n"; \
	__s << "ERROR : " << getErrorDesc(__status) << "\n" ; \
	__s << "<== " #stmt "\n"; \
	LOG_ERROR( __s.str() ); \
	} \
}

/*
* OpenCLKernel constructor
*/
OpenCLKernel::OpenCLKernel( int platform, int device ) : GPUKernel(),
   m_hContext(0),
   m_hQueue(0),
   m_dRandoms(0),
	m_dBitmap(0), 
   m_dTextures(0),
	m_dPrimitives(0), 
   m_dBoxPrimitivesIndex(0), 
   m_dPostProcessingBuffer(0),
   m_preferredWorkGroupSize(0)
{
	int  status(0);
	cl_platform_id   platforms[MAX_DEVICES];
	cl_uint          ret_num_devices;
	cl_uint          ret_num_platforms;
	char buffer[MAX_SOURCE_SIZE];
	size_t len;

#ifdef LOGGING
	// Initialize Log
	LOG_INITIALIZE_ETW(
		&GPU_OPENCLRAYTRACERMODULE,
		&GPU_OPENCLRAYTRACERMODULE_EVENT_DEBUG,
		&GPU_OPENCLRAYTRACERMODULE_EVENT_VERBOSE,
		&GPU_OPENCLRAYTRACERMODULE_EVENT_INFO, 
		&GPU_OPENCLRAYTRACERMODULE_EVENT_WARNING,
		&GPU_OPENCLRAYTRACERMODULE_EVENT_ERROR);
#endif // NDEBUG

#if USE_KINECT
	// Initialize Kinect
	status = NuiInitialize( NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);

	m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

	m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
	status = NuiSkeletonTrackingEnable( m_skeletons, 0 );

	status = NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
	status = NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

	status = NuiCameraElevationSetAngle( 0 );
#endif // USE_KINECT

	std::stringstream s;
   s << "--------------------------------------------------------------------------------\n";
	LOG_INFO("clGetPlatformIDs\n");
	CHECKSTATUS(clGetPlatformIDs(MAX_DEVICES, platforms, &ret_num_platforms));
   s << ret_num_platforms << " platorm(s) detected" << "\n";

	//for( int p=0;p<ret_num_platforms;++p)
   int p=platform;
	{
		// Platform details
		s << "Platform " << p << ":\n";
		CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_PROFILE, MAX_SOURCE_SIZE, buffer, &len ));
		buffer[len] = 0; s << "  Profile    : " << buffer << "\n";
		CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_VERSION, MAX_SOURCE_SIZE, buffer, &len ));
		buffer[len] = 0; s << "  Version    : " << buffer << "\n";
		CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_NAME, MAX_SOURCE_SIZE, buffer, &len ));
		buffer[len] = 0; s << "  Name       : " << buffer << "\n";
		CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_VENDOR, MAX_SOURCE_SIZE, buffer, &len ));
		buffer[len] = 0; s << "  Vendor     : " << buffer << "\n";
		CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_EXTENSIONS, MAX_SOURCE_SIZE, buffer, &len ));
		buffer[len] = 0; s << "  Extensions : " << buffer << "\n";

		CHECKSTATUS(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 1, m_hDevices, &ret_num_devices));

		// Devices
		//for( int d=0; d<ret_num_devices; ++d)
      int d = device;
		{
			s << "  Device " << d << ":\n";

			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			s << "    DEVICE_NAME                        : " << buffer << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
			s << "    DEVICE_VENDOR                      : " << buffer << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
			s << "    DEVICE_VERSION                     : " << buffer << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
			s << "    DRIVER_VERSION                     : " << buffer << "\n";

			cl_uint value;
			cl_uint values[10];
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, NULL));
			s << "    DEVICE_MAX_COMPUTE_UNITS           : " << value << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), &value, NULL));
			s << "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : " << value << "\n";
			//CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(value), &value, NULL));
			//s << "    CL_DEVICE_MAX_WORK_GROUP_SIZE      : " << value << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(values), &values, NULL));
			s << "    CL_DEVICE_MAX_WORK_ITEM_SIZES      : " << values[0] << ", " << values[1] << ", " << values[2] << "\n";
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), &value, NULL));
			s << "    CL_DEVICE_MAX_CLOCK_FREQUENCY      : " << value << "\n";


			cl_device_type infoType;
			CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL));
			s << "    DEVICE_TYPE                        : ";
			if (infoType & CL_DEVICE_TYPE_DEFAULT) {
				infoType &= ~CL_DEVICE_TYPE_DEFAULT;
				s << "Default";
			}
			if (infoType & CL_DEVICE_TYPE_CPU) {
				infoType &= ~CL_DEVICE_TYPE_CPU;
				s << "CPU";
			}
			if (infoType & CL_DEVICE_TYPE_GPU) {
				infoType &= ~CL_DEVICE_TYPE_GPU;
				s << "GPU";
			}
			if (infoType & CL_DEVICE_TYPE_ACCELERATOR) {
				infoType &= ~CL_DEVICE_TYPE_ACCELERATOR;
				s << "Accelerator";
			}
			if (infoType != 0) {
				s << "Unknown ";
				s << infoType;
			}
		}
		s << "\n";
	}
   s << "--------------------------------------------------------------------------------\n";
	LOG_INFO( s.str() );

	m_hContext = clCreateContext(NULL, ret_num_devices, &m_hDevices[0], NULL, NULL, &status );
	m_hQueue = clCreateCommandQueue(m_hContext, m_hDevices[0], CL_QUEUE_PROFILING_ENABLE, &status);

	// Eye position
	m_viewPos.x =   0.0f;
	m_viewPos.y =   0.0f;
	m_viewPos.z = -40.0f;

	// Rotation angles
	m_angles.x = 0.0f;
	m_angles.y = 0.0f;
	m_angles.z = 0.0f;
}

/*
* compileKernels
*/
void OpenCLKernel::compileKernels( 
	const KernelSourceType sourceType,
	const std::string& source, 
	const std::string& ptxFileName,
	const std::string& options)
{
	int status(0);
	cl_program hProgram(0);
	try 
	{
		clUnloadCompiler();

		const char* source_str; 
		size_t len(0);
		switch( sourceType ) 
		{
		case kst_file:
			if( source.length() != 0 ) 
			{
				source_str = loadFromFile(source, len);
			}
			break;
		case kst_string:
			{
				source_str = source.c_str();
				len = source.length();
			}
			break;
		}

      //saveToFile("encoded.cl", source_str );

		LOG_INFO("clCreateProgramWithSource\n");
		hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&source_str, (const size_t*)&len, &status );
		CHECKSTATUS(status);

		LOG_INFO("clCreateProgramWithSource\n");
		hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&source_str, (const size_t*)&len, &status );
		CHECKSTATUS(status);

		LOG_INFO("clBuildProgram\n");
		CHECKSTATUS( clBuildProgram( hProgram, 0, NULL, options.c_str(), NULL, NULL) );

		if( sourceType == kst_file)
		{
			delete [] source_str;
			source_str = NULL;
		}

		clUnloadCompiler();

		LOG_INFO("clCreateKernel(render_kernel)\n");
		m_kStandardRenderer = clCreateKernel( hProgram, "k_standardRenderer", &status );
		CHECKSTATUS(status);

		LOG_INFO("clCreateKernel(postProcessing_kernel)\n");
      m_kDefault = clCreateKernel( hProgram, "k_default", &status );
		CHECKSTATUS(status);

      cl_int computeUnits;
		clGetKernelWorkGroupInfo( m_kStandardRenderer, m_hDevices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(cl_int), &computeUnits , NULL);
		std::cout << "CL_KERNEL_WORK_GROUP_SIZE=" << computeUnits << std::endl;

		clGetKernelWorkGroupInfo( m_kStandardRenderer, m_hDevices[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(m_preferredWorkGroupSize), &m_preferredWorkGroupSize , NULL);
		std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=" << m_preferredWorkGroupSize << std::endl;

		char buffer[MAX_SOURCE_SIZE];
		LOG_INFO("clGetProgramBuildInfo\n");
		CHECKSTATUS( clGetProgramBuildInfo( hProgram, m_hDevices[0], CL_PROGRAM_BUILD_LOG, MAX_SOURCE_SIZE*sizeof(char), &buffer, &len ) );

		if( buffer[0] != 0 ) 
		{
			buffer[len] = 0;
			std::stringstream s;
			s << buffer;
			LOG_INFO( s.str() );
			std::cout << s.str() << std::endl;
		}

#if 0
		// Generate Binaries!!!
		// Obtain the length of the binary data that will be queried, for each device
		size_t ret_num_devices = 1;
		size_t binaries_sizes[MAX_DEVICES];
		CHECKSTATUS( clGetProgramInfo(
			hProgram, 
			CL_PROGRAM_BINARY_SIZES, 
			ret_num_devices*sizeof(size_t), 
			binaries_sizes, 
			0 ));

		char **binaries = new char*[MAX_DEVICES];
		for (size_t i = 0; i < ret_num_devices; i++)
			binaries[i] = new char[binaries_sizes[i]+1];

		CHECKSTATUS( clGetProgramInfo(
			hProgram, 
			CL_PROGRAM_BINARIES, 
			MAX_DEVICES*sizeof(size_t), 
			binaries, 
			NULL));                        

		for (size_t i = 0; i < ret_num_devices; i++) {
			binaries[i][binaries_sizes[i]] = '\0';
			char name[255];
			sprintf_s(name, 255, "kernel%d.ptx", i );
			FILE* fp = NULL;
			fopen_s(&fp, name, "w");
			fwrite(binaries[i], 1, binaries_sizes[i], fp);
			fclose(fp);
		}

		for (size_t i = 0; i < ret_num_devices; i++)                                
			delete [] binaries[i];                        
		delete [] binaries;
#endif // 0

		if( ptxFileName.length() != 0 ) 
		{
			// Open the ptx file and load it   
			// into a char* buffer   
			std::ifstream myReadFile;
			std::string str;
			std::string line;
			std::ifstream myfile( ptxFileName.c_str() );
			if (myfile.is_open()) {
				while ( myfile.good() ) {
					std::getline(myfile,line);
					str += '\n' + line;
				}
				myfile.close();
			}

			size_t lSize = str.length();
			char* buffer = new char[lSize+1];
			strcpy_s( buffer, lSize, str.c_str() );

			// Build the rendering kernel
			int errcode(0);
			hProgram = clCreateProgramWithBinary(
				m_hContext,
				1, 
				&m_hDevices[0],
				&lSize, 
				(const unsigned char**)&buffer,                 
				&status, 
				&errcode);   
			CHECKSTATUS(errcode);

			CHECKSTATUS( clBuildProgram( hProgram, 0, NULL, "", NULL, NULL) );

		   LOG_INFO("clGetProgramBuildInfo\n");
		   CHECKSTATUS( clGetProgramBuildInfo( hProgram, m_hDevices[0], CL_PROGRAM_BUILD_LOG, MAX_SOURCE_SIZE*sizeof(char), &buffer, &lSize ) );

		   if( buffer[0] != 0 ) 
		   {
			   buffer[lSize] = 0;
			   std::stringstream s;
			   s << buffer;
			   LOG_INFO( s.str() );
			   std::cout << s.str() << std::endl;
		   }

			m_kStandardRenderer = clCreateKernel(
				hProgram, "render_kernel", &status );
			CHECKSTATUS(status);

			delete [] buffer;
		}

		LOG_INFO("clReleaseProgram\n");
		CHECKSTATUS(clReleaseProgram(hProgram));
		hProgram = 0;
	}
	catch( ... ) 
	{
		LOG_ERROR("Unexpected exception\n");
	}
}

void OpenCLKernel::initializeDevice()
{
	int status(0);
	// Setup device memory
	LOG_INFO("Setup device memory\n");
   int size = m_sceneInfo.width.x*m_sceneInfo.height.x;
   m_dBitmap       = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, size*sizeof(char)*gColorDepth,      0, NULL);
	m_dDepthOfField = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, size*sizeof(float4),             0, NULL);

	m_dRandoms             = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY,  size*sizeof(float),       0, NULL);
	m_dPostProcessingBuffer= clCreateBuffer( m_hContext, CL_MEM_READ_ONLY,  size*sizeof(float4),      0, NULL);

   m_dBoundingBoxes       = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(BoundingBox)*NB_MAX_BOXES,    0, NULL);
   m_dPrimitives          = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Primitive)*NB_MAX_PRIMITIVES, 0, NULL);
   m_dBoxPrimitivesIndex  = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(int)*NB_MAX_PRIMITIVES,       0, NULL);
   m_dLamps               = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(int)*NB_MAX_LAMPS,            0, NULL);

   m_dMaterials  = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Material)*NB_MAX_MATERIALS, 0, NULL);

   m_dTextures   = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gTextureWidth*gTextureHeight*gTextureDepth*sizeof(char)*NB_MAX_TEXTURES, 0, NULL);

#if USE_KINECT
	m_dVideo      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gVideoWidth*gVideoHeight*gKinectColorVideo, 0, NULL);
	m_dDepth      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gDepthWidth*gDepthHeight*gKinectColorDepth, 0, NULL);
#endif // USE_KINECT
}

void OpenCLKernel::releaseDevice()
{
	LOG_INFO("Release device memory\n");
	if( m_dPrimitives ) CHECKSTATUS(clReleaseMemObject(m_dPrimitives));
	if( m_dBoundingBoxes ) CHECKSTATUS(clReleaseMemObject(m_dBoundingBoxes));
	if( m_dBoxPrimitivesIndex ) CHECKSTATUS(clReleaseMemObject(m_dBoxPrimitivesIndex));
	if( m_dMaterials ) CHECKSTATUS(clReleaseMemObject(m_dMaterials));
	if( m_dTextures ) CHECKSTATUS(clReleaseMemObject(m_dTextures));
	if( m_dRandoms ) CHECKSTATUS(clReleaseMemObject(m_dRandoms));
   if( m_dPostProcessingBuffer ) CHECKSTATUS(clReleaseMemObject(m_dPostProcessingBuffer));
	if( m_dBitmap ) CHECKSTATUS(clReleaseMemObject(m_dBitmap));
	if( m_dDepthOfField ) CHECKSTATUS(clReleaseMemObject(m_dDepthOfField));
	if( m_kStandardRenderer ) CHECKSTATUS(clReleaseKernel(m_kStandardRenderer));
   if( m_kDefault ) CHECKSTATUS(clReleaseKernel(m_kDefault));
	if( m_hQueue ) CHECKSTATUS(clReleaseCommandQueue(m_hQueue));
	if( m_hContext ) CHECKSTATUS(clReleaseContext(m_hContext));
}

/*
* runKernel
*/
void OpenCLKernel::render_begin( const float timer )
{
	int status(0);

#if USE_KINECT
	char* video(0);
	char* depth(0);
	// Video
	const NUI_IMAGE_FRAME* pImageFrame = 0;
	WaitForSingleObject (m_hNextVideoFrameEvent,INFINITE); 
	status = NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame ); 
	if(( status == S_OK) && pImageFrame ) 
	{
		INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
		NUI_LOCKED_RECT LockedRect;
		pTexture->LockRect( 0, &LockedRect, NULL, 0 ) ; 
		if( LockedRect.Pitch != 0 ) 
		{
			video = (char*) LockedRect.pBits;
		}
	}

	// Depth
	const NUI_IMAGE_FRAME* pDepthFrame = 0;
	WaitForSingleObject (m_hNextDepthFrameEvent,INFINITE); 
	status = NuiImageStreamGetNextFrame( m_pDepthStreamHandle, 0, &pDepthFrame ); 
	if(( status == S_OK) && pDepthFrame ) 
	{
		INuiFrameTexture* pTexture = pDepthFrame->pFrameTexture;
		if( pTexture ) 
		{
			NUI_LOCKED_RECT LockedRectDepth;
			pTexture->LockRect( 0, &LockedRectDepth, NULL, 0 ) ; 
			if( LockedRectDepth.Pitch != 0 ) 
			{
				depth = (char*) LockedRectDepth.pBits;
			}
		}
	}
	NuiImageStreamReleaseFrame( m_pVideoStreamHandle, pImageFrame ); 
	NuiImageStreamReleaseFrame( m_pDepthStreamHandle, pDepthFrame ); 
#endif // USE_KINECT

	// Initialise Input arrays
   CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dBoundingBoxes, CL_TRUE, 0, (m_nbActivePrimitives+1)*sizeof(BoundingBox), m_hBoundingBoxes, 0, NULL, NULL));
   CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dBoxPrimitivesIndex, CL_TRUE, 0, (m_nbActivePrimitives+1)*sizeof(int), m_hBoxPrimitivesIndex, 0, NULL, NULL));
	CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dPrimitives, CL_TRUE, 0, (m_nbActivePrimitives+1)*sizeof(Primitive), m_hPrimitives, 0, NULL, NULL));
	CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dMaterials,  CL_TRUE, 0, (m_nbActiveMaterials+1)*sizeof(Material), m_hMaterials,  0, NULL, NULL));
	CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dLamps, CL_TRUE, 0, (m_nbActiveLamps+1)*sizeof(int), m_hLamps,  0, NULL, NULL));

	if( !m_texturedTransfered )
	{
		CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dTextures,   CL_TRUE, 0, gTextureDepth*gTextureWidth*gTextureHeight*m_nbActiveTextures, m_hTextures,   0, NULL, NULL));
		CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dRandoms,    CL_TRUE, 0, m_sceneInfo.width.x*m_sceneInfo.height.x*sizeof(float), m_hRandoms,    0, NULL, NULL));
		m_texturedTransfered = true;
	}

#ifdef USE_KINECT
	if( video ) CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hVideo, CL_TRUE, 0, gKinectColorVideo*gVideoWidth*gVideoHeight, video, 0, NULL, NULL));
	if( depth ) CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hDepth, CL_TRUE, 0, gKinectColorDepth*gDepthWidth*gDepthHeight, depth, 0, NULL, NULL));
#endif // USE_KINECT

   // Define scene parameters
   Ray ray;
   memset(&ray, 0, sizeof(Ray));
   ray.origin    = m_viewPos;
   ray.direction = m_viewDir;

	// Run the raytracing kernel!!
	CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 0, sizeof(cl_mem), (void*)&m_dBoundingBoxes ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 1, sizeof(cl_int), (void*)&m_nbActiveBoxes ));
	CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 2, sizeof(cl_mem), (void*)&m_dBoxPrimitivesIndex ));
	CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 3, sizeof(cl_mem), (void*)&m_dPrimitives ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 4, sizeof(cl_int), (void*)&m_nbActivePrimitives ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 5, sizeof(cl_mem), (void*)&m_dLamps ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 6, sizeof(cl_int), (void*)&m_nbActiveLamps ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 7, sizeof(cl_mem), (void*)&m_dMaterials ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 8, sizeof(cl_mem), (void*)&m_dTextures ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 9, sizeof(cl_mem), (void*)&m_dRandoms ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,10, sizeof(Ray),               (void*)&ray ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,11, sizeof(cl_float4),         (void*)&m_angles ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,12, sizeof(SceneInfo),         (void*)&m_sceneInfo ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,13, sizeof(cl_float),          (void*)&timer ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,14, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
   CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,15, sizeof(cl_mem),            (void*)&m_dPostProcessingBuffer ));

   CHECKSTATUS(clSetKernelArg( m_kDefault, 0, sizeof(SceneInfo),         (void*)&m_sceneInfo ));
   CHECKSTATUS(clSetKernelArg( m_kDefault, 1, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
   CHECKSTATUS(clSetKernelArg( m_kDefault, 2, sizeof(cl_mem),            (void*)&m_dPostProcessingBuffer ));
   CHECKSTATUS(clSetKernelArg( m_kDefault, 3, sizeof(cl_mem),            (void*)&m_dBitmap ));

	// Run the post processing kernel!!
   size_t szGlobalWorkSize[] = { m_sceneInfo.width.x, m_sceneInfo.height.x };
	CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kStandardRenderer, 2, NULL, szGlobalWorkSize, 0, 0, 0, 0));

	// Post Processing
	CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kDefault, 2, NULL, szGlobalWorkSize, 0, 0, 0, 0));
}
  
void OpenCLKernel::render_end( char* bitmap)
{
	// ------------------------------------------------------------
	// Read back the results
	// ------------------------------------------------------------
	CHECKSTATUS( clEnqueueReadBuffer( m_hQueue, m_dBitmap, CL_TRUE, 0, m_sceneInfo.width.x*m_sceneInfo.height.x*sizeof(char)*gColorDepth, bitmap, 0, NULL, NULL) );

	CHECKSTATUS(clFlush(m_hQueue));
	CHECKSTATUS(clFinish(m_hQueue));
}

void OpenCLKernel::initBuffers()
{
   GPUKernel::initBuffers();
	initializeDevice();
   compileKernels( kst_file, "resource.dll", "", "-cl-fast-relaxed-math" );
}

OpenCLKernel::~OpenCLKernel()
{
	// Clean up
	releaseDevice();

#if USE_KINECT
	CloseHandle(m_skeletons);
	CloseHandle(m_hNextDepthFrameEvent); 
	CloseHandle(m_hNextVideoFrameEvent); 
	CloseHandle(m_hNextSkeletonEvent);
	NuiShutdown();
#endif // USE_KINECT
}

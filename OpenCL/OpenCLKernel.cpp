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

#include <GL/freeglut.h>

#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "OpenCLKernel.h"
#include "../Logging.h"

const long MAX_SOURCE_SIZE = 3*65535;
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
   LOG_ERROR( "OpenCL Error (via pfn_notify): " << user_data );
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
	LOG_ERROR( __s.str().c_str() ); \
	} \
}

/*
* OpenCLKernel constructor
*/
OpenCLKernel::OpenCLKernel( bool activeLogging, int optimalNbOfPrimmitivesPerBox, int selectedPlatform, int selectedDevice )
 : GPUKernel( activeLogging, optimalNbOfPrimmitivesPerBox ),
   m_platform(selectedPlatform),
   m_device(selectedDevice),
   m_hContext(0),
   m_hQueue(0),
   m_dRandoms(0),
	m_dBitmap(0), 
   m_dTextures(0),
	m_dPrimitives(0), 
   m_dPostProcessingBuffer(0),
   m_dPrimitivesXYIds(0),
   m_dLamps(0),
   m_dLightInformation(0),
   m_preferredWorkGroupSize(0),
   m_kStandardRenderer(0),
   m_kAnaglyphRenderer(0),
   m_k3DVisionRenderer(0),
   m_kFishEyeRenderer(0),
   m_kDefault(0),
   m_kDepthOfField(0),
   m_kAmbientOcclusion(0),
   m_kRadiosity(0)
{
	int  status(0);
	cl_platform_id   platforms[MAX_DEVICES];
	cl_device_id     devices[MAX_DEVICES];
	cl_uint          ret_num_devices;
	cl_uint          ret_num_platforms;

   // TODO: Occupancy parameters
   m_occupancyParameters.x = 1;
   m_occupancyParameters.y = 1;

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
   LOG_INFO(1,"--------------------------------------------------------------------------------");
	LOG_INFO(3,"clGetPlatformIDs");
	CHECKSTATUS(clGetPlatformIDs(MAX_DEVICES, platforms, &ret_num_platforms));
   LOG_INFO(1,"Number of platforms detected: " << ret_num_platforms);

#if 1
	for( cl_uint platform(0);platform<ret_num_platforms;++platform)
   {
		// Platform details
      LOG_INFO(1,"----------------------------------------");
		LOG_INFO(1, "Platform " << platform );
      LOG_INFO(1,"----------------------------------------");
		CHECKSTATUS(clGetPlatformInfo( platforms[platform], CL_PLATFORM_PROFILE, MAX_SOURCE_SIZE, buffer, &len )); buffer[len] = 0; 
      LOG_INFO(1, "  Profile    : " << buffer);
		CHECKSTATUS(clGetPlatformInfo( platforms[platform], CL_PLATFORM_VERSION, MAX_SOURCE_SIZE, buffer, &len )); buffer[len] = 0; 
      LOG_INFO(1, "  Version    : " << buffer);
		CHECKSTATUS(clGetPlatformInfo( platforms[platform], CL_PLATFORM_NAME, MAX_SOURCE_SIZE, buffer, &len )); buffer[len] = 0; 
      LOG_INFO(1, "  Name       : " << buffer);
		CHECKSTATUS(clGetPlatformInfo( platforms[platform], CL_PLATFORM_VENDOR, MAX_SOURCE_SIZE, buffer, &len )); buffer[len] = 0; 
      LOG_INFO(1, "  Vendor     : " << buffer);
		CHECKSTATUS(clGetPlatformInfo( platforms[platform], CL_PLATFORM_EXTENSIONS, MAX_SOURCE_SIZE, buffer, &len )); buffer[len] = 0; 
      LOG_INFO(1, "  Extensions : " << buffer);

		CHECKSTATUS(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &ret_num_devices));

		// Devices
		for( cl_uint device=0; device<ret_num_devices; ++device)
      {
         LOG_INFO(1,"   -------------------------------------");
         if( platform==m_platform && device==m_device )
         {
            LOG_INFO(1,"   Device " << device << " --- Selected platform and device");
            m_hDeviceId = devices[device];
	         m_hContext = clCreateContext(NULL, ret_num_devices, &devices[0], NULL, NULL, &status );
            CHECKSTATUS(status);
	         m_hQueue = clCreateCommandQueue(m_hContext, m_hDeviceId, NULL, &status);
            CHECKSTATUS(status);
         }
         else
         {
   		   LOG_INFO(1,"   Device " << device);
         }
         LOG_INFO(1,"   -------------------------------------");

			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			LOG_INFO(1,"    DEVICE_NAME                        : " << buffer);
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
			LOG_INFO(1,"    DEVICE_VENDOR                      : " << buffer);
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
			LOG_INFO(1,"    DEVICE_VERSION                     : " << buffer);
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
			LOG_INFO(1,"    DRIVER_VERSION                     : " << buffer);

			cl_uint value;
			cl_uint values[10];
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, NULL));
			LOG_INFO(1,"    DEVICE_MAX_COMPUTE_UNITS           : " << value);
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), &value, NULL));
			LOG_INFO(1,"    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : " << value);
			//CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(value), &value, NULL));
			//LOG_INFO(1,"    CL_DEVICE_MAX_WORK_GROUP_SIZE      : " << value << "\n";
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(values), &values, NULL));
			LOG_INFO(1,"    CL_DEVICE_MAX_WORK_ITEM_SIZES      : " << values[0] << ", " << values[1] << ", " << values[2]);
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), &value, NULL));
			LOG_INFO(1,"    CL_DEVICE_MAX_CLOCK_FREQUENCY      : " << value);
         
			cl_device_type infoType;
			CHECKSTATUS(clGetDeviceInfo(devices[device], CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL));
			LOG_INFO(1,"    DEVICE_TYPE                        : ");
			if (infoType & CL_DEVICE_TYPE_DEFAULT) 
         {
				infoType &= ~CL_DEVICE_TYPE_DEFAULT;
				LOG_INFO(1,"      Default");
			}
			if (infoType & CL_DEVICE_TYPE_CPU) 
         {
				infoType &= ~CL_DEVICE_TYPE_CPU;
				LOG_INFO(1,"      CPU");
			}
			if (infoType & CL_DEVICE_TYPE_GPU) 
         {
				infoType &= ~CL_DEVICE_TYPE_GPU;
				LOG_INFO(1,"      GPU");
			}
			if (infoType & CL_DEVICE_TYPE_ACCELERATOR) 
         {
				infoType &= ~CL_DEVICE_TYPE_ACCELERATOR;
				LOG_INFO(1,"      Accelerator");
			}
			if (infoType != 0) 
         {
				LOG_INFO(1,"      Unknown " << infoType);
         }

		}
	}
   LOG_INFO(1,"--------------------------------------------------------------------------------");
#else
	CHECKSTATUS(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 1, m_hDevices, &ret_num_devices));
#endif // 0

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
	LOG_INFO(3,"OpenCLKernel::compileKernels");
	int status(0);
	cl_program hProgram(0);
	try 
	{
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

		LOG_INFO(1,"clCreateProgramWithSource");
		hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&source_str, (const size_t*)&len, &status );
		CHECKSTATUS(status);

		LOG_INFO(1,"clBuildProgram");
      status = clBuildProgram(hProgram, 0, NULL, options.c_str(), NULL, NULL);
      if( status!=CL_SUCCESS )
      {
         size_t length;
		   char buffer[MAX_SOURCE_SIZE];
         memset(buffer,0,MAX_SOURCE_SIZE);
         CHECKSTATUS(clGetProgramBuildInfo(hProgram, m_hDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length));
         LOG_ERROR("Program Build failed: " << buffer );
      }

		if( sourceType == kst_file)
		{
			delete [] source_str;
			source_str = NULL;
		}

      // Rendering kernels
		LOG_INFO(1,"Creating Rendering kernels");
		m_kStandardRenderer = clCreateKernel( hProgram, "k_standardRenderer", &status );
		CHECKSTATUS(status);

      m_kAnaglyphRenderer = clCreateKernel( hProgram, "k_anaglyphRenderer", &status );
		CHECKSTATUS(status);

      m_k3DVisionRenderer = clCreateKernel( hProgram, "k_3DVisionRenderer", &status );
		CHECKSTATUS(status);

      m_kFishEyeRenderer = clCreateKernel( hProgram, "k_fishEyeRenderer", &status );
		CHECKSTATUS(status);

      // Post-processing kernels
		LOG_INFO(1,"Creating Post-processing kernels");
      m_kDefault = clCreateKernel( hProgram, "k_default", &status );
		CHECKSTATUS(status);

      m_kDepthOfField = clCreateKernel( hProgram, "k_depthOfField", &status );
		CHECKSTATUS(status);

      m_kAmbientOcclusion = clCreateKernel( hProgram, "k_ambientOcclusion", &status );
		CHECKSTATUS(status);

      m_kRadiosity = clCreateKernel( hProgram, "k_radiosity", &status );
		CHECKSTATUS(status);

      cl_int computeUnits;
		clGetKernelWorkGroupInfo( m_kStandardRenderer, m_hDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(cl_int), &computeUnits , NULL);
		LOG_INFO(1,"CL_KERNEL_WORK_GROUP_SIZE=" << computeUnits );

		clGetKernelWorkGroupInfo( m_kStandardRenderer, m_hDeviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(m_preferredWorkGroupSize), &m_preferredWorkGroupSize , NULL);
		LOG_INFO(1,"CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=" << m_preferredWorkGroupSize );

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
#ifdef WIN32
			strcpy_s( buffer, lSize, str.c_str() );
#else
			strcpy( buffer, str.c_str() );
#endif // WIN32

			// Build the rendering kernel
			int errcode(0);
			hProgram = clCreateProgramWithBinary(
				m_hContext,
				1, 
				&m_hDeviceId,
				&lSize, 
				(const unsigned char**)&buffer,                 
				&status, 
				&errcode);   
			CHECKSTATUS(errcode);

			CHECKSTATUS( clBuildProgram( hProgram, 0, NULL, "", NULL, NULL) );

		   LOG_INFO(3,"clGetProgramBuildInfo");
		   CHECKSTATUS( clGetProgramBuildInfo( hProgram, m_hDeviceId, CL_PROGRAM_BUILD_LOG, MAX_SOURCE_SIZE*sizeof(char), &buffer, &lSize ) );

		   if( buffer[0] != 0 ) 
		   {
			   buffer[lSize] = 0;
			   std::stringstream s;
			   s << buffer;
			   LOG_INFO(3, s.str().c_str() );
			   LOG_INFO(3,s.str() );
		   }

			delete [] buffer;
		}

		LOG_INFO(3,"clReleaseProgram");
		CHECKSTATUS(clReleaseProgram(hProgram));
		hProgram = 0;
	}
	catch( ... ) 
	{
		LOG_ERROR("Unexpected exception");
	}
}

void OpenCLKernel::initializeDevice()
{
	LOG_INFO(3,"OpenCLKernel::initializeDevice");
	int status(0);
	// Setup device memory
	LOG_INFO(3,"Setup device memory");
   int size = m_sceneInfo.width.x*m_sceneInfo.height.x;
   m_dBitmap              = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, size*sizeof(BitmapBuffer)*gColorDepth,   0, NULL);
	m_dRandoms             = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY,  size*sizeof(RandomBuffer),         0, NULL);
   m_dPostProcessingBuffer= clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, size*sizeof(PostProcessingBuffer), 0, NULL);
   m_dPrimitivesXYIds     = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, size*sizeof(PrimitiveXYIdBuffer),  0, NULL);
   m_dBoundingBoxes       = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(BoundingBox)*NB_MAX_BOXES,                  0, NULL);
   m_dPrimitives          = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Primitive)*NB_MAX_PRIMITIVES,               0, NULL);
   m_dLamps               = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Lamp)*NB_MAX_LAMPS,                         0, NULL);
   m_dLightInformation    = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(LightInformation)*NB_MAX_LIGHTINFORMATIONS, 0, NULL);
   m_dMaterials           = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Material)*NB_MAX_MATERIALS, 0, NULL);

#if USE_KINECT
	m_dVideo      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gVideoWidth*gVideoHeight*gKinectColorVideo, 0, NULL);
	m_dDepth      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gDepthWidth*gDepthHeight*gKinectColorDepth, 0, NULL);
#endif // USE_KINECT
}

void OpenCLKernel::releaseDevice()
{
	LOG_INFO(3,"Release device memory");
	if( m_dPrimitives ) CHECKSTATUS(clReleaseMemObject(m_dPrimitives));
	if( m_dBoundingBoxes ) CHECKSTATUS(clReleaseMemObject(m_dBoundingBoxes));
	if( m_dMaterials ) CHECKSTATUS(clReleaseMemObject(m_dMaterials));
	if( m_dTextures ) CHECKSTATUS(clReleaseMemObject(m_dTextures));
	if( m_dRandoms ) CHECKSTATUS(clReleaseMemObject(m_dRandoms));
   if( m_dPostProcessingBuffer ) CHECKSTATUS(clReleaseMemObject(m_dPostProcessingBuffer));
   if( m_dPrimitivesXYIds ) CHECKSTATUS(clReleaseMemObject(m_dPrimitivesXYIds));
	if( m_dBitmap ) CHECKSTATUS(clReleaseMemObject(m_dBitmap));

   // Rendering kernels
   if( m_kStandardRenderer ) CHECKSTATUS(clReleaseKernel(m_kStandardRenderer));
   if( m_kAnaglyphRenderer ) CHECKSTATUS(clReleaseKernel(m_kAnaglyphRenderer));
   if( m_k3DVisionRenderer ) CHECKSTATUS(clReleaseKernel(m_k3DVisionRenderer));
   if( m_kFishEyeRenderer )  CHECKSTATUS(clReleaseKernel(m_kFishEyeRenderer));

   // Post processing kernels
	if( m_kDefault )          CHECKSTATUS(clReleaseKernel(m_kDefault));
	if( m_kDepthOfField )     CHECKSTATUS(clReleaseKernel(m_kDepthOfField));
	if( m_kAmbientOcclusion ) CHECKSTATUS(clReleaseKernel(m_kAmbientOcclusion));
	if( m_kRadiosity )      CHECKSTATUS(clReleaseKernel(m_kRadiosity));

   // Queue and context
   if( m_hQueue ) CHECKSTATUS(clReleaseCommandQueue(m_hQueue));
	if( m_hContext ) CHECKSTATUS(clReleaseContext(m_hContext));
}

/*
* runKernel
*/
void OpenCLKernel::render_begin( const float timer )
{
   GPUKernel::render_begin(timer);
	int status(0);
   if( m_refresh )
   {
	   // CPU -> GPU Data transfers
      int nbBoxes      = m_nbActiveBoxes[m_frame];
      int nbPrimitives = m_nbActivePrimitives[m_frame];
      int nbLamps      = m_nbActiveLamps[m_frame];
      int nbMaterials  = m_nbActiveMaterials+1;
      LOG_INFO(3, "Data sizes [" << m_frame << "]: " << nbBoxes << ", " << nbPrimitives << ", " << nbMaterials << ", " << nbLamps );
         
      if( !m_primitivesTransfered )
      {
         CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dBoundingBoxes,      CL_TRUE, 0, nbBoxes*sizeof(BoundingBox),                     m_hBoundingBoxes,      0, NULL, NULL));
	      CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dPrimitives,         CL_TRUE, 0, nbPrimitives*sizeof(Primitive),                  m_hPrimitives,         0, NULL, NULL));
	      CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dLamps,              CL_TRUE, 0, nbLamps*sizeof(Lamp),                            m_hLamps,              0, NULL, NULL));
	      CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dLightInformation,   CL_TRUE, 0, m_lightInformationSize*sizeof(LightInformation), m_lightInformation,    0, NULL, NULL));
         m_primitivesTransfered = true;
      }
	
      if( !m_materialsTransfered )
	   {
         realignTexturesAndMaterials();

         CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dRandoms,   CL_TRUE, 0, m_sceneInfo.width.x*m_sceneInfo.height.x*sizeof(RandomBuffer), m_hRandoms,      0, NULL, NULL));
         CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dMaterials, CL_TRUE, 0, nbMaterials*sizeof(Material),   m_hMaterials,    0, NULL, NULL));
		   m_materialsTransfered = true;
         LOG_INFO(1, "Transfering " << nbMaterials << " materials");
	   }

#ifdef USE_KINECT
   if( m_kinectEnabled )
   {
	   // Video
	   const NUI_IMAGE_FRAME* pImageFrame = 0;
	   WaitForSingleObject (m_hNextVideoFrameEvent,INFINITE); 
	   HRESULT status = NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame ); 
	   if(( status == S_OK) && pImageFrame ) 
	   {
		   INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
		   NUI_LOCKED_RECT LockedRect;
		   pTexture->LockRect( 0, &LockedRect, NULL, 0 ) ; 
		   if( LockedRect.Pitch != 0 ) 
		   {
			   m_hVideo = (unsigned char*)LockedRect.pBits;
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
				   m_hDepth = (unsigned char*)LockedRectDepth.pBits;
			   }
		   }
	   }
	   NuiImageStreamReleaseFrame( m_pVideoStreamHandle, pImageFrame ); 
	   NuiImageStreamReleaseFrame( m_pDepthStreamHandle, pDepthFrame ); 

      m_hTextures[0].buffer = m_hVideo;
      m_hTextures[1].buffer = m_hDepth;
   }

      if( !m_texturesTransfered )
	   {
         LOG_INFO(1, "Transfering " << m_nbActiveTextures << " textures, and " << m_lightInformationSize << " light information");
         h2d_textures( 
            m_occupancyParameters,
            NB_MAX_TEXTURES,  m_hTextures );
		      m_texturesTransfered = true;
      }
#endif // USE_KINECT

      if( !m_texturesTransfered )
	   {
         LOG_INFO(1, "Transfering " << m_nbActiveTextures << " textures, and " << m_lightInformationSize << " light information");
         int totalSize(0);
         for( int i(0); i<m_nbActiveTextures; ++i )
         {
            totalSize += m_hTextures[i].size.x*m_hTextures[i].size.y*m_hTextures[i].size.z;
         }
      	
         if( m_dTextures ) 
         {
            LOG_INFO(1, "Releasing existing texture resources" );
            CHECKSTATUS(clReleaseMemObject(m_dTextures));
            m_dTextures = 0;
         }

         if( totalSize>0 )
         {
            // Allocate host buffer
            BitmapBuffer* tmpTextures = new BitmapBuffer[totalSize];
            for( int i(0); i<m_nbActiveTextures; ++i )
            {
               if( m_hTextures[i].buffer != nullptr )
               {
                  int textureSize = m_hTextures[i].size.x*m_hTextures[i].size.y*m_hTextures[i].size.z;
                  memcpy(tmpTextures+m_hTextures[i].offset,m_hTextures[i].buffer,textureSize);
               }
            }
            m_dTextures = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , totalSize*sizeof(BitmapBuffer), 0, NULL);
            CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_dTextures, CL_TRUE, 0, totalSize*sizeof(BitmapBuffer), tmpTextures, 0, NULL, NULL));
            LOG_INFO( 1, "Total GPU texture memory allocated: " << totalSize << " bytes" );
            delete tmpTextures;
         }
         m_texturesTransfered = true;
      }

      // Kernel execution
      LOG_INFO(3, "CPU PostProcessingBuffer: " << sizeof(PostProcessingBuffer));
      LOG_INFO(3, "CPU PrimitiveXYIdBuffer : " << sizeof(PrimitiveXYIdBuffer));
      LOG_INFO(3, "CPU BoundingBox         : " << sizeof(BoundingBox));
      LOG_INFO(3, "CPU Primitive           : " << sizeof(Primitive));
      LOG_INFO(3, "CPU Material            : " << sizeof(Material));

      SceneInfo sceneInfo=m_sceneInfo;
      if( m_sceneInfo.pathTracingIteration.x==0 ) sceneInfo.graphicsLevel.x = 1;
      if( m_sceneInfo.pathTracingIteration.x==m_sceneInfo.maxPathTracingIterations.x-1 ) sceneInfo.misc.w = 2; // Antialiasing on last iteration

      size_t szLocalWorkSize[] = { 1, 1 };
      size_t szGlobalWorkSize[] = { m_sceneInfo.width.x/szLocalWorkSize[0], m_sceneInfo.height.x/szLocalWorkSize[1] };
      int zero(0);
	   switch( sceneInfo.renderingType.x ) 
	   {
	   case vtAnaglyph:
		   {
	         // Run the post processing kernel!!
	         CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 1, sizeof(cl_int),   (void*)&zero )); 
	         CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 2, sizeof(cl_int),   (void*)&zero ));
	         CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 3, sizeof(cl_mem),   (void*)&m_dBoundingBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 4, sizeof(cl_int),   (void*)&nbBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 5, sizeof(cl_mem),   (void*)&m_dPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 6, sizeof(cl_int),   (void*)&nbPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 7, sizeof(cl_mem),   (void*)&m_dLightInformation ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 8, sizeof(cl_int),   (void*)&m_lightInformationSize ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer, 9, sizeof(cl_int),   (void*)&nbLamps ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,10, sizeof(cl_mem),   (void*)&m_dMaterials ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,11, sizeof(cl_mem),   (void*)&m_dTextures ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,12, sizeof(cl_mem),   (void*)&m_dRandoms ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,13, sizeof(cl_float3),(void*)&m_viewPos ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,14, sizeof(cl_float3),(void*)&m_viewDir ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,15, sizeof(cl_float3),(void*)&m_angles ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,16, sizeof(SceneInfo),(void*)&sceneInfo ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,17, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,18, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
            CHECKSTATUS(clSetKernelArg( m_kAnaglyphRenderer,19, sizeof(cl_mem),   (void*)&m_dPrimitivesXYIds ));
	         CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kAnaglyphRenderer, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
			   break;
		   }
	   case vt3DVision:
		   {
	         CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 1, sizeof(cl_int),   (void*)&zero )); 
	         CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 2, sizeof(cl_int),   (void*)&zero ));
	         CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 3, sizeof(cl_mem),   (void*)&m_dBoundingBoxes ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 4, sizeof(cl_int),   (void*)&nbBoxes ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 5, sizeof(cl_mem),   (void*)&m_dPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 6, sizeof(cl_int),   (void*)&nbPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 7, sizeof(cl_mem),   (void*)&m_dLightInformation ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 8, sizeof(cl_int),   (void*)&m_lightInformationSize ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer, 9, sizeof(cl_int),   (void*)&nbLamps ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,10, sizeof(cl_mem),   (void*)&m_dMaterials ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,11, sizeof(cl_mem),   (void*)&m_dTextures ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,12, sizeof(cl_mem),   (void*)&m_dRandoms ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,13, sizeof(cl_float3),(void*)&m_viewPos ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,14, sizeof(cl_float3),(void*)&m_viewDir ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,15, sizeof(cl_float3),(void*)&m_angles ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,16, sizeof(SceneInfo),(void*)&sceneInfo ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,17, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,18, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
            CHECKSTATUS(clSetKernelArg( m_k3DVisionRenderer,19, sizeof(cl_mem),   (void*)&m_dPrimitivesXYIds ));
	         CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_k3DVisionRenderer, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
			   break;
		   }
	   case vtFishEye:
		   {
	         CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 1, sizeof(cl_int),   (void*)&zero )); 
	         CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 2, sizeof(cl_int),   (void*)&zero ));
	         CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 3, sizeof(cl_mem),   (void*)&m_dBoundingBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 4, sizeof(cl_int),   (void*)&nbBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 5, sizeof(cl_mem),   (void*)&m_dPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 6, sizeof(cl_int),   (void*)&nbPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 7, sizeof(cl_mem),   (void*)&m_dLightInformation ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 8, sizeof(cl_int),   (void*)&m_lightInformationSize ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer, 9, sizeof(cl_int),   (void*)&nbLamps ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,10, sizeof(cl_mem),   (void*)&m_dMaterials ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,11, sizeof(cl_mem),   (void*)&m_dTextures ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,12, sizeof(cl_mem),   (void*)&m_dRandoms ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,13, sizeof(cl_float3),(void*)&m_viewPos ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,14, sizeof(cl_float3),(void*)&m_viewDir ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,15, sizeof(cl_float3),(void*)&m_angles ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,16, sizeof(SceneInfo),(void*)&sceneInfo ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,17, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,18, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
            CHECKSTATUS(clSetKernelArg( m_kFishEyeRenderer,19, sizeof(cl_mem),   (void*)&m_dPrimitivesXYIds ));
	         CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kFishEyeRenderer, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
			   break;
		   }
	   default:
		   {
	         CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 1, sizeof(cl_int),   (void*)&zero )); 
	         CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 2, sizeof(cl_int),   (void*)&zero ));
	         CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 3, sizeof(cl_mem),   (void*)&m_dBoundingBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 4, sizeof(cl_int),   (void*)&nbBoxes ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 5, sizeof(cl_mem),   (void*)&m_dPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 6, sizeof(cl_int),   (void*)&nbPrimitives ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 7, sizeof(cl_mem),   (void*)&m_dLightInformation ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 8, sizeof(cl_int),   (void*)&m_lightInformationSize ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer, 9, sizeof(cl_int),   (void*)&nbLamps ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,10, sizeof(cl_mem),   (void*)&m_dMaterials ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,11, sizeof(cl_mem),   (void*)&m_dTextures ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,12, sizeof(cl_mem),   (void*)&m_dRandoms ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,13, sizeof(cl_float3),(void*)&m_viewPos ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,14, sizeof(cl_float3),(void*)&m_viewDir ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,15, sizeof(cl_float3),(void*)&m_angles ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,16, sizeof(SceneInfo),(void*)&sceneInfo ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,17, sizeof(PostProcessingInfo),(void*)&m_postProcessingInfo ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,18, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
            CHECKSTATUS(clSetKernelArg( m_kStandardRenderer,19, sizeof(cl_mem),   (void*)&m_dPrimitivesXYIds ));
	         CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kStandardRenderer, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
			   break;
		   }
      }

      // --------------------------------------------------------------------------------
      // Post processing
      // --------------------------------------------------------------------------------
	   switch( m_postProcessingInfo.type.x )
	   {
	   case ppe_depthOfField:
         CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
         CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 1, sizeof(SceneInfo),(void*)&sceneInfo ));
         CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 2, sizeof(PostProcessingInfo),   (void*)&m_postProcessingInfo ));
	      CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 3, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
	      CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 4, sizeof(cl_mem),   (void*)&m_dRandoms ));
         CHECKSTATUS(clSetKernelArg( m_kDepthOfField, 5, sizeof(cl_mem),   (void*)&m_dBitmap ));
	      CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kDepthOfField, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
		   break;
	   case ppe_ambientOcclusion:
         CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
         CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 1, sizeof(SceneInfo),(void*)&sceneInfo ));
         CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 2, sizeof(PostProcessingInfo),   (void*)&m_postProcessingInfo ));
	      CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 3, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
	      CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 4, sizeof(cl_mem),   (void*)&m_dRandoms ));
         CHECKSTATUS(clSetKernelArg( m_kAmbientOcclusion, 5, sizeof(cl_mem),   (void*)&m_dBitmap ));
	      CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kAmbientOcclusion, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
		   break;
	   case ppe_radiosity:
         CHECKSTATUS(clSetKernelArg( m_kRadiosity, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
         CHECKSTATUS(clSetKernelArg( m_kRadiosity, 1, sizeof(SceneInfo),(void*)&sceneInfo ));
         CHECKSTATUS(clSetKernelArg( m_kRadiosity, 2, sizeof(PostProcessingInfo),   (void*)&m_postProcessingInfo ));
         CHECKSTATUS(clSetKernelArg( m_kRadiosity, 3, sizeof(cl_mem),   (void*)&m_dPrimitivesXYIds ));
	      CHECKSTATUS(clSetKernelArg( m_kRadiosity, 4, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
	      CHECKSTATUS(clSetKernelArg( m_kRadiosity, 5, sizeof(cl_mem),   (void*)&m_dRandoms ));
         CHECKSTATUS(clSetKernelArg( m_kRadiosity, 6, sizeof(cl_mem),   (void*)&m_dBitmap ));
	      CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kRadiosity, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
		   break;
	   case ppe_oneColor:
         // TODO!
         break;
	   default:
         CHECKSTATUS(clSetKernelArg( m_kDefault, 0, sizeof(cl_int2),  (void*)&m_occupancyParameters ));
         CHECKSTATUS(clSetKernelArg( m_kDefault, 1, sizeof(SceneInfo),(void*)&sceneInfo ));
         CHECKSTATUS(clSetKernelArg( m_kDefault, 2, sizeof(PostProcessingInfo),   (void*)&m_postProcessingInfo ));
	      CHECKSTATUS(clSetKernelArg( m_kDefault, 3, sizeof(cl_mem),   (void*)&m_dPostProcessingBuffer ));
         CHECKSTATUS(clSetKernelArg( m_kDefault, 4, sizeof(cl_mem),   (void*)&m_dBitmap ));
	      CHECKSTATUS(clEnqueueNDRangeKernel( m_hQueue, m_kDefault, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, 0));
		   break;
	   }
   }
   m_refresh = (m_sceneInfo.pathTracingIteration.x<m_sceneInfo.maxPathTracingIterations.x);
}
  
void OpenCLKernel::render_end()
{
	// ------------------------------------------------------------
	// Read back the results
	// ------------------------------------------------------------
	CHECKSTATUS( clEnqueueReadBuffer( m_hQueue, m_dBitmap,          CL_FALSE, 0, m_sceneInfo.width.x*m_sceneInfo.height.x*sizeof(BitmapBuffer)*gColorDepth, m_bitmap, 0, NULL, NULL) );
   CHECKSTATUS( clEnqueueReadBuffer( m_hQueue, m_dPrimitivesXYIds, CL_FALSE, 0, m_sceneInfo.width.x*m_sceneInfo.height.x*sizeof(PrimitiveXYIdBuffer), m_hPrimitivesXYIds, 0, NULL, NULL) );
   
	CHECKSTATUS(clFlush(m_hQueue));
	CHECKSTATUS(clFinish(m_hQueue));
   
   if( m_sceneInfo.misc.x == 0 )
   {
      ::glEnable(GL_TEXTURE_2D);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      ::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      ::glTexImage2D(GL_TEXTURE_2D, 0, 3, m_sceneInfo.width.x, m_sceneInfo.height.x, 0, GL_RGB, GL_UNSIGNED_BYTE, m_bitmap);

      if( m_sceneInfo.renderingType.x == vt3DVision )
      {
         float step = 0.1f;
         float halfStep = 1.f;
         float scale = 2.f;
         float distortion=1.f;

         for( int a(0); a<2; ++a ) 
         {
            float2 center = {0.f,0.f};
            center.x = (a==0) ? -0.5f: 0.5f;
            float b = (a==0) ? 0.f : 0.5f;

            for( float x(0); x<1; x+=step )
            {
               for( float y(0); y<1; y+=step )
               {

                  float2 s;
                  s.x = scale;
                  s.y = scale;

                  float2 p0 = {s.x*x-halfStep,        s.y*y-halfStep};
                  float2 p1 = {s.x*(x+step)-halfStep, s.y*y-halfStep};
                  float2 p2 = {s.x*(x+step)-halfStep, s.y*(y+step)-halfStep};
                  float2 p3 = {s.x*x-halfStep,        s.y*(y+step)-halfStep};

                  float d0 = sqrt(pow(p0.x,2)+pow(p0.y,2));
                  float d1 = sqrt(pow(p1.x,2)+pow(p1.y,2));
                  float d2 = sqrt(pow(p2.x,2)+pow(p2.y,2));
                  float d3 = sqrt(pow(p3.x,2)+pow(p3.y,2));

                  d0 = 1.f-pow(d0,2.f)*m_distortion;
                  d1 = 1.f-pow(d1,2.f)*m_distortion;
                  d2 = 1.f-pow(d2,2.f)*m_distortion;
                  d3 = 1.f-pow(d3,2.f)*m_distortion;

                  /*
                  d0 = (d0==0.f) ? 1.f : (1.f/(sin(d0-M_PI/2.f)+m_distortion));
                  d1 = (d1==0.f) ? 1.f : (1.f/(sin(d1-M_PI/2.f)+m_distortion));
                  d2 = (d2==0.f) ? 1.f : (1.f/(sin(d2-M_PI/2.f)+m_distortion));
                  d3 = (d3==0.f) ? 1.f : (1.f/(sin(d3-M_PI/2.f)+m_distortion));
                  */

                  ::glBegin(GL_QUADS);
                  ::glTexCoord2f(1.f-(b+(x/2.f)), y);
                  ::glVertex3f(center.x+0.5f*p0.x*d0, center.y+p0.y*d0, 0.f);

                  ::glTexCoord2f(1.f-(b+(x+step)/2.f), y);
                  ::glVertex3f(center.x+0.5f*p1.x*d1, center.y+p1.y*d1, 0.f);

                  ::glTexCoord2f(1.f-(b+(x+step)/2.f), y+step);
                  ::glVertex3f(center.x+0.5f*p2.x*d2, center.y+p2.y*d2, 0.f);

                  ::glTexCoord2f(1.f-(b+(x/2.f)), y+step);
                  ::glVertex3f(center.x+0.5f*p3.x*d3, center.y+p3.y*d3, 0.f);
                  ::glEnd();
               }
            }
         }
      }
      else
      {
         ::glBegin(GL_QUADS);
         ::glTexCoord2f(1.f,0.f);
         ::glVertex3f(-1.f, -1.f, 0.f);

         ::glTexCoord2f(0.f,0.f);
         ::glVertex3f( 1.f, -1.f, 0.f);

         ::glTexCoord2f(0.f,1.f);
         ::glVertex3f( 1.f,  1.f, 0.f);

         ::glTexCoord2f(1.f,1.f);
         ::glVertex3f(-1.f,  1.f, 0.f);
         ::glEnd();
      }
      ::glDisable(GL_TEXTURE_2D);
   }
}

void OpenCLKernel::initBuffers()
{
	LOG_INFO(3,"OpenCLKernel::initBuffers");
   GPUKernel::initBuffers();
	initializeDevice();
   compileKernels( kst_file, "RayTracer.cl", "", "-cl-fast-relaxed-math" );
}

OpenCLKernel::~OpenCLKernel()
{
	LOG_INFO(3,"OpenCLKernel::~OpenCLKernel");
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

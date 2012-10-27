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

#pragma once

#include <CL/opencl.h>
#include <vector_types.h>


#include <stdio.h>
#include <string>
#include <windows.h>
#if USE_KINECT
#include <nuiapi.h>
#endif // USE_KINECT

#include "../Consts.h"
#include "../GPUKernel.h"

class RAYTRACINGENGINE_API OpenCLKernel : public GPUKernel
{

public:
   enum KernelSourceType
   {
	   kst_file,
	   kst_string
   };

public:
	OpenCLKernel( int platform = 0, int device = 0 );
	~OpenCLKernel();

   virtual void initBuffers();

public:
	// ---------- Devices ----------
	void initializeDevice();
	void releaseDevice();

	void compileKernels(
		const KernelSourceType sourceType,
		const std::string& source, 
		const std::string& ptxFileName,
		const std::string& options);

public:
	// ---------- Rendering ----------
	void render_begin( const float timer );
   void render_end( char* bitmap);

private:
	// OpenCL Objects
	cl_device_id     m_hDevices[100];
	int              m_hPlatformId;
	cl_context       m_hContext;
	cl_command_queue m_hQueue;
	cl_kernel        m_kStandardRenderer;
	cl_kernel        m_kDefault;
	cl_uint          m_preferredWorkGroupSize;

private:

	cl_mem m_dBoundingBoxes;
   cl_mem m_dPrimitives;
   cl_mem m_dBoxPrimitivesIndex;
	cl_mem m_dLamps;
	cl_mem m_dMaterials;
	cl_mem m_dTextures;
	cl_mem m_dDepthOfField;
	cl_mem m_dRandoms;
	cl_mem m_dBitmap;
   cl_mem m_dPostProcessingBuffer;

#ifdef KINECT
private:
   cl_mem m_dVideo;
	cl_mem m_dDepth;
#endif // KINECT

};

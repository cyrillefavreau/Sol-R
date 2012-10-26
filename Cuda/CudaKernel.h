/* 
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

#ifdef WIN32
	#include <windows.h>
#endif

#include "../DLL_API.h"
#include "../GPUKernel.h"

#if USE_KINECT
#include <nuiapi.h>
#endif // USE_KINECT

class RAYTRACINGENGINE_API CudaKernel : public GPUKernel
{
public:

   CudaKernel();
	~CudaKernel();

   virtual void initBuffers();

public:
	// ---------- Devices ----------
	void initializeDevice();
	void releaseDevice();

   void deviceQuery();

   void resetBoxesAndPrimitives();

public:
	// ---------- Rendering ----------
	void render_begin( const float timer );
   void render_end( char* bitmap);

public:

   void setBlockSize( int x, int y, int z)    { m_blockSize.x = x; m_blockSize.y = y; m_blockSize.z = z; };
   void setSharedMemSize( int sharedMemSize ) { m_sharedMemSize = sharedMemSize; };

private:

   // Runtime kernel execution parameters
   dim3 m_blockSize;
   int  m_sharedMemSize;

};

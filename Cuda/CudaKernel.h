/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "../DLL_API.h"
#include "../GPUKernel.h"

class RAYTRACINGENGINE_API CudaKernel : public GPUKernel
{
public:

   CudaKernel(bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device );
	~CudaKernel();

   virtual void initBuffers();
   virtual void cleanup();

public:

   void recompileKernels(const std::string& kernelCode="");

public:
	// ---------- Devices ----------
	void initializeDevice();
	void releaseDevice();

   virtual void reshape();

   void deviceQuery();

   void resetBoxesAndPrimitives();

public:
	// ---------- Rendering ----------
	void render_begin( const float timer );
   void render_end();

public:
   
   virtual std::string getGPUDescription();

public:

   void setBlockSize( int x, int y, int z)    { m_blockSize.x = x; m_blockSize.y = y; m_blockSize.z = z; };
   void setSharedMemSize( int sharedMemSize ) { m_sharedMemSize = sharedMemSize; };

private:

   // Runtime kernel execution parameters
   int4 m_blockSize;
   int  m_sharedMemSize;

};

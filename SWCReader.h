/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <map>

#include "GPUKernel.h"

struct Morphology
{
   int branch;
   float x;
   float y;
   float z;
   float radius;
   int parent;
   int primitiveId;
};
typedef std::map<int, Morphology> Morphologies;

class RAYTRACINGENGINE_API SWCReader
{
public:
   SWCReader();
   ~SWCReader();

   CPUBoundingBox loadMorphologyFromFile( 
      const std::string& filename,
      GPUKernel& cudaKernel,
      const Vertex& center,
      const bool autoScale,
      const Vertex& scale, 
      bool autoCenter,
      const int materialId );

   Morphologies getMorphologies() { return m_morphologies; }

private:
   Morphologies m_morphologies;

};

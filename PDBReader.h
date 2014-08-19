/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "GPUKernel.h"

enum GeometryType
{
   gtAtoms           = 0,
   gtFixedSizeAtoms  = 1,
   gtSticks          = 2,
   gtAtomsAndSticks  = 3,
   gtIsoSurface      = 4,
   gtBackbone        = 5
};

class RAYTRACINGENGINE_API PDBReader
{
public:

   PDBReader(void);
   virtual ~PDBReader(void);

public:

   Vertex loadAtomsFromFile( 
      const std::string& filename,
      GPUKernel&   cudaKernel,
      GeometryType geometryType,
      const float  defaultAtomSize,
      const float  defaultStickSize,
      const int    materialType,
      const Vertex scale,
      const bool   useModels = false);

   int getNbBoxes() { return m_nbBoxes; }
   int getNbPrimitives() { return m_nbPrimitives; }

private:

   int m_nbPrimitives;
   int m_nbBoxes;

};

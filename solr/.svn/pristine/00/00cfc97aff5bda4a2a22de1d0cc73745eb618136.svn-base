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

   float3 loadAtomsFromFile( 
      const std::string& filename,
      GPUKernel&   cudaKernel,
      GeometryType geometryType,
      const float  defaultAtomSize,
      const float  defaultStickSize,
      const int    materialType,
      const float3 scale,
      const bool   useModels = false);

   int getNbBoxes() { return m_nbBoxes; }
   int getNbPrimitives() { return m_nbPrimitives; }

private:

   int m_nbPrimitives;
   int m_nbBoxes;

};

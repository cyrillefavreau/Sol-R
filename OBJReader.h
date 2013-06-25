/* 
* Raytracing Engine
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

#include <map>

#include "GPUKernel.h"

struct MaterialMTL
{
   unsigned int index;
   float3 Ka;
   float3 Kd;
   float3 Ks;
   float  Ns;
   float  reflection;
   float  transparency;
   float  refraction;
};

class RAYTRACINGENGINE_API OBJReader
{
public:
   OBJReader(void);
   ~OBJReader(void);

   unsigned int OBJReader::loadMaterialsFromFile(
      const std::string& filename,
      std::map<std::string,MaterialMTL>& m_materials,
      GPUKernel& GPUKernel,
      int materialId);

   float3 loadModelFromFile( 
      const int& frame,
      const std::string& filename,
      GPUKernel& cudaKernel,
      const float3& center,
      const bool autoScale,
      const float& scale, 
      int materialId);

};

/* 
* Copyright (C) 2011-2014 Cyrille Favreau <cyrille_favreau@hotmail.com>
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
   Vertex Ka;
   Vertex Kd;
   Vertex Ks;
   float  Ns;
   float  reflection;
   float  transparency;
   float  refraction;
   float  noise;
   int    diffuseTextureId;
   int    normalTextureId;
   int    bumpTextureId;
   int    specularTextureId;
   float  illumination;
   bool   isSketchupLightMaterial;
};

class RAYTRACINGENGINE_API OBJReader
{
public:
   OBJReader();
   ~OBJReader();

   unsigned int loadMaterialsFromFile(
      const std::string& filename,
      std::map<std::string,MaterialMTL>& m_materials,
      GPUKernel& GPUKernel,
      int materialId);

   Vertex loadModelFromFile( 
      const std::string& filename,
      GPUKernel& cudaKernel,
      const Vertex& center,
      const bool autoScale,
      const Vertex& scale, 
      bool loadMaterials,
      int  materialId,
      bool allSpheres);

private:
   void addLightComponent(
      GPUKernel& kernel,
      std::vector<Vertex>& solrVertices,
      const Vertex& center,
      const Vertex& objectCenter,
      const Vertex& objectScale,
      const int material);
};

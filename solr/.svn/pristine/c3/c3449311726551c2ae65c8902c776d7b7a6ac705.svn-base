/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
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
   float  opacity;
   float  refraction;
   float  noise;
   int    diffuseTextureId;
   int    normalTextureId;
   int    bumpTextureId;
   int    specularTextureId;
   int    reflectionTextureId;
   int    transparencyTextureId;
   int    ambientOcclusionTextureId;
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
      bool allSpheres,
      bool autoCenter,
      CPUBoundingBox& aabb,
      const bool& checkInAABB,
      const CPUBoundingBox& inAABB);

private:
   void addLightComponent(
      GPUKernel& kernel,
      std::vector<Vertex>& solrVertices,
      const Vertex& center,
      const Vertex& objectCenter,
      const Vertex& objectScale,
      const int material,
      CPUBoundingBox& aabb);
};

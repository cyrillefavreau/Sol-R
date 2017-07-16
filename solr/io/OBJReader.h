/* Copyright (c) 2011-2014, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of Sol-R <https://github.com/cyrillefavreau/Sol-R>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <map>

#include "../GPUKernel.h"

struct MaterialMTL {
  unsigned int index;
  Vertex Ka;
  Vertex Kd;
  Vertex Ks;
  float Ns;
  float reflection;
  float transparency;
  float opacity;
  float refraction;
  float noise;
  int diffuseTextureId;
  int normalTextureId;
  int bumpTextureId;
  int specularTextureId;
  int reflectionTextureId;
  int transparencyTextureId;
  int ambientOcclusionTextureId;
  float illumination;
  bool isSketchupLightMaterial;
};

class RAYTRACINGENGINE_API OBJReader {
public:
  OBJReader();
  ~OBJReader();

  unsigned int
  loadMaterialsFromFile(const std::string &filename,
                        std::map<std::string, MaterialMTL> &m_materials,
                        GPUKernel &GPUKernel, int materialId);

  Vertex loadModelFromFile(const std::string &filename, GPUKernel &cudaKernel,
                           const Vertex &center, const bool autoScale,
                           const Vertex &scale, bool loadMaterials,
                           int materialId, bool allSpheres, bool autoCenter,
                           CPUBoundingBox &aabb, const bool &checkInAABB,
                           const CPUBoundingBox &inAABB);

private:
  void addLightComponent(GPUKernel &kernel, std::vector<Vertex> &solrVertices,
                         const Vertex &center, const Vertex &objectCenter,
                         const Vertex &objectScale, const int material,
                         CPUBoundingBox &aabb);
};
